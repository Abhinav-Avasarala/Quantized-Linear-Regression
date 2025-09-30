import torch

# ===== 1) Setup =====
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

m, n = 10000, 5000
W  = torch.randn(m, n, device=device) * 10
b  = torch.randn(m,     device=device) * 10
x0 = torch.zeros(n,     device=device)    # zero init -> min-norm solution if underdetermined

def objective(x):
    r = W @ x - b
    return 0.5 * (r @ r)  # 0.5 * ||r||²

def grad(x):
    return W.T @ (W @ x - b)

#--------------------
#----- Quantization params ---
#--------------------
bits_w = 4                         # weight bits (keep at 4 like your original)
bits_x = 4                         # activation bits (you can try 8 for better accuracy)
max_w  = (2 ** (bits_w - 1)) - 1   # 7 for 4-bit
max_x  = (2 ** (bits_x - 1)) - 1   # 7 for 4-bit
tiny   = 1e-12

#--------------------
#----- Row-wise symmetric quantization for W ---
#--------------------
# Per-row scale so that max |W[i,:]| maps to max_w
s_row = (W.abs().amax(dim=1) / max_w).clamp_min(1e-8)                    # [m]
W_q   = torch.round((W.T / s_row).T).clamp(-max_w, max_w).to(torch.int8) # [m,n] int8

def Wx_q(x):
    """
    Approximate y = W @ x using quantized W only:
      W ≈ diag(s_row) * W_q  =>  y ≈ diag(s_row) * (W_q @ x)
    """
    y_int = (W_q.float() @ x)     # placeholder; real speed needs int8 kernels (int32 acc)
    return s_row * y_int

def WT_r_q(r):
    """
    Approximate g = W^T @ r using quantized W only:
      W^T ≈ W_q^T * diag(s_row)  =>  g ≈ W_q^T @ (s_row * r)
    """
    r_scaled = s_row * r
    g_int = (W_q.float().T @ r_scaled)
    return g_int

#--------------------
#----- Quantize vector x (per-tensor symmetric) ---
#--------------------
def quantize_vec(x, maximum=max_x):
    """
    s_x = max(|x|)/maximum  (clamped),  x_q = round(x/s_x) clipped to [-maximum, maximum]
    x ≈ s_x * x_q
    """
    s_x = (x.abs().amax() / maximum).clamp_min(1e-8)
    x_q = torch.round(x / s_x).clamp(-maximum, maximum).to(torch.int8)
    return s_x, x_q

def Wx_Wq_xq(x):
    """
    Approximate y = W @ x with BOTH W and x quantized:
      W ≈ diag(s_row) * W_q ,  x ≈ s_x * x_q
      => y ≈ s_x * diag(s_row) * (W_q @ x_q)
    """
    s_x, x_q = quantize_vec(x, maximum=max_x)
    y_int = (W_q.float() @ x_q.float())    # placeholder; real int8 GEMV would be faster
    return (s_row * y_int) * s_x

#--------------------
#----- Stepsize (Lipschitz) estimates ---
#--------------------
def estimate_sigma_max_sq(A, iters=20):
    """
    Power iteration estimate of sigma_max(A)^2 for a *linear* A (float matrix or linear op).
    """
    y = torch.randn(A.shape[1], device=A.device)
    y = y / (y.norm() + tiny)
    for _ in range(iters):
        y = A.T @ (A @ y)
        y = y / (y.norm() + tiny)
    Ay = A @ y
    return float(Ay @ Ay)

# Full-precision L
L_est = estimate_sigma_max_sq(W, iters=25)
eta = 0.9 / L_est

# For W-only quantized ops you originally used a quantized power method; keep it:
def power_L_est_q(t=25, n=W.shape[1], device=W.device):
    y = torch.randn(n, device=device); y /= (y.norm() + tiny)
    for _ in range(t):
        z = WT_r_q(Wx_q(y))     # ≈ (W^T W) y
        y = z / (z.norm() + tiny)
    Az = Wx_q(y)
    return float(Az @ Az)

Lq   = power_L_est_q(t=25)
eta_q = 0.9 / Lq

# For BOTH W & x quantized, avoid a nonlinear power method (x's scale depends on y).
# Instead, build a linear float proxy:  W_tilde = diag(s_row) @ W_q (as float).
W_tilde = (W_q.float().T * s_row).T
Lq_both = estimate_sigma_max_sq(W_tilde, iters=25)
eta_both = 0.9 / Lq_both

# Guard against pathological estimates
if not (torch.isfinite(torch.tensor(eta_both)) and eta_both > 1e-12):
    eta_both = max(eta_q, eta)

# ===== 2) Closed-form (least-norm LS) =====
x_closed = torch.linalg.lstsq(W, b).solution
f_closed = objective(x_closed)
res_closed = torch.linalg.norm(W @ x_closed - b)

# ===== 3) Run the three GD variants =====
T = 10000

x       = x0.clone()                  # Full-precision GD iterate
x_qW    = torch.zeros(n, device=device)  # GD with quantized W only
x_both  = torch.zeros(n, device=device)  # GD with both W and x quantized

for _ in range(T):
    # -- Full-precision GD --
    x -= eta * grad(x)

    # -- GD with quantized W only --
    r_qW = Wx_q(x_qW) - b
    g_qW = WT_r_q(r_qW)
    x_qW -= eta_q * g_qW

    # -- GD with BOTH W and x quantized --
    r_b  = Wx_Wq_xq(x_both) - b
    g_b  = WT_r_q(r_b)              # W^T r ≈ W_q^T (s_row * r)
    x_both -= eta_both * g_b

# ===== 3.5) Metrics vs closed-form x* only (evaluate with TRUE W) =====
f_gd      = objective(x)
res_gd    = torch.linalg.norm(W @ x      - b)

f_qW      = objective(x_qW)
res_qW    = torch.linalg.norm(W @ x_qW   - b)

f_both    = objective(x_both)
res_both  = torch.linalg.norm(W @ x_both - b)

def report_vs_closed(name, xcand, f_cand, res_cand):
    rel_err     = torch.linalg.norm(xcand - x_closed) / (torch.linalg.norm(x_closed) + tiny)
    rel_obj     = abs(f_cand - f_closed) / (abs(f_closed) + tiny)
    cos_sim     = torch.dot(xcand, x_closed) / ((torch.norm(xcand) + tiny) * (torch.norm(x_closed) + tiny))
    rel_res_gap = (res_cand - res_closed) / (res_closed + tiny)
    print(f"{name:20s}: rel ‖x−x*‖ = {rel_err.item():.2e} , "
          f"cos(x, x*) = {cos_sim.item():.6f} , "
          f"rel residual gap = {rel_res_gap.item():.2e} , "
          f"rel obj diff = {rel_obj.item():.2e}")

print(f"Device: {device}")
print(f"Problem size: W={m}x{n}")
print(f"Stepsizes: eta(full) = {eta:.3e} , eta(W-quant) = {eta_q:.3e} , eta(W+X-quant) = {eta_both:.3e}")
print(f"Iterations = {T}")
print("------------------------------------------------------")
print(f"Closed-form:          residual = {res_closed.item():.6f} , f(x*)      = {f_closed.item():.6f}")
print(f"Gradient descent:     residual = {res_gd.item():.6f} , f(x_T)     = {f_gd.item():.6f}")
print(f"Quantized-W GD:       residual = {res_qW.item():.6f} , f(x_qW)    = {f_qW.item():.6f}")
print(f"Quantized W+X GD:     residual = {res_both.item():.6f} , f(x_both) = {f_both.item():.6f}")
print("------------------------------------------------------")
print("All comparisons are vs closed-form x*")
report_vs_closed("Full-prec GD",       x,      f_gd,   res_gd)
report_vs_closed("Quantized-W GD",     x_qW,   f_qW,   res_qW)
report_vs_closed("Quantized W+X",      x_both, f_both, res_both)

# rel_err = ‖x_candidate − x_closed‖₂ / ‖x_closed‖₂
# cos_sim = (x_candidate · x_closed) / (‖x_candidate‖₂ · ‖x_closed‖₂)
# rel_res_gap = (‖W x_candidate − b‖₂ − ‖W x_closed − b‖₂) / ‖W x_closed − b‖₂
# rel_obj = | f(x_candidate) − f(x_closed) | / f(x_closed)
