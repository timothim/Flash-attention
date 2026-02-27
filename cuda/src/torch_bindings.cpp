/**
 * torch_bindings.cpp — PyTorch C++ extension bindings for the Flash
 * Attention CUDA kernels.
 *
 * Exposes three Python-callable functions:
 *   flash_attn_cuda.naive_fwd(Q, K, V, causal) -> O
 *   flash_attn_cuda.flash_fwd(Q, K, V, causal) -> (O, L)
 *   flash_attn_cuda.flash_bwd(Q, K, V, O, dO, L, causal) -> (dQ, dK, dV)
 *
 * All inputs must be contiguous FP16 tensors on CUDA with shape
 * [B, H, N, D].
 */

#include <torch/extension.h>
#include <cuda_fp16.h>
#include "flash_attn.cuh"

// ── Helpers ─────────────────────────────────────────────────────────────

#define CHECK_CUDA(x) \
    TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FP16(x) \
    TORCH_CHECK((x).dtype() == torch::kFloat16, #x " must be float16")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FP16(x)

static half* ptr(torch::Tensor& t) {
    return reinterpret_cast<half*>(t.data_ptr<at::Half>());
}
static const half* cptr(const torch::Tensor& t) {
    return reinterpret_cast<const half*>(t.data_ptr<at::Half>());
}

// ── Naive forward ───────────────────────────────────────────────────────

torch::Tensor naive_fwd(torch::Tensor Q, torch::Tensor K,
                        torch::Tensor V, bool causal) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    auto O = torch::zeros_like(Q);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    naive_attention_fwd(cptr(Q), cptr(K), cptr(V), ptr(O),
                        B, H, N, D, causal, stream);
    return O;
}

// ── Flash forward ───────────────────────────────────────────────────────

std::vector<torch::Tensor> flash_fwd(torch::Tensor Q, torch::Tensor K,
                                     torch::Tensor V, bool causal) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    auto O = torch::zeros_like(Q);
    auto L = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    flash_attn_fwd(cptr(Q), cptr(K), cptr(V), ptr(O),
                   L.data_ptr<float>(),
                   B, H, N, D, causal, stream);

    return {O, L};
}

// ── Flash backward ──────────────────────────────────────────────────────

std::vector<torch::Tensor> flash_bwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO, torch::Tensor L,
    bool causal)
{
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_INPUT(O); CHECK_INPUT(dO);
    TORCH_CHECK(L.is_cuda() && L.is_contiguous() && L.dtype() == torch::kFloat32,
                "L must be a contiguous float32 CUDA tensor");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    flash_attn_bwd(cptr(Q), cptr(K), cptr(V), cptr(O), cptr(dO),
                   L.data_ptr<float>(),
                   ptr(dQ), ptr(dK), ptr(dV),
                   B, H, N, D, causal, stream);

    return {dQ, dK, dV};
}

// ── PyTorch module registration ─────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_fwd", &naive_fwd,
          "Naive attention forward (materialises N×N score matrix)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("causal") = false);
    m.def("flash_fwd", &flash_fwd,
          "Flash Attention 2 forward",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("causal") = false);
    m.def("flash_bwd", &flash_bwd,
          "Flash Attention 2 backward",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("dO"), py::arg("L"),
          py::arg("causal") = false);
}
