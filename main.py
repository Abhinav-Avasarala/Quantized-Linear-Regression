import torch

# ===== 1) Setup =====
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

m, n = 10000, 5000
W  = torch.randn(m, n, device=device) * 10
b  = torch.randn(m,     device=device) * 10
x0 = torch.zeros(n,     device=device)    # <-- zero init to reach min-norm solution

def objective(x):
    r = W @ x - b
    return 0.5 * (r @ r) # 0.5 * ||r||²
#r @ r = dot product of the residual with itself = sum of squared residuals = |r|^2_2


def grad(x):
    return W.T @ (W @ x - b)

#--------------------
#----- Minimal Pytorch quantization --- 
#--------------------
bits = 4
maximum = (2 ** (bits-1)) - 1
eps = 1e-12
s_row = (W.abs().amax(dim=1) / maximum).clamp_min(1e-8)      # [m]
W_q   = torch.round((W.T / s_row).T).clamp(-maximum, maximum).to(torch.int8)  # [m,n] int8

def Wx_q(x):
    # y = W @ x ≈ diag(s_row) * (W_q @ x)
    # For speed you'd want int8 GEMV to int32 then scale; PyTorch fallback uses float matmul
    y_int = (W_q.float() @ x)           # placeholder; real speed needs int8 kernels
    return s_row * y_int

def WT_r_q(r):
    # g = W^T @ r ≈ W_q^T @ (s_row * r)
    r_scaled = s_row * r
    g_int = (W_q.float().T @ r_scaled)
    return g_int

# --- GD with quantized W ---
def power_L_est_q(matvec, t=25, n=W.shape[1], device=W.device):
    y = torch.randn(n, device=device); y /= (y.norm() + 1e-12)
    for _ in range(t):
        z = WT_r_q(Wx_q(y))             # effectively (W^T W) y using quantized ops
        y = z / (z.norm() + 1e-12)
    Az = Wx_q(y)
    return float(Az @ Az)               # ~ sigma_max^2

Lq = power_L_est_q(Wx_q, t=25)
eta_q = 0.9 / Lq

#-----------------------
#----------------------



# ===== 2) Closed-form (least-norm LS) =====
x_closed = torch.linalg.lstsq(W, b).solution
f_closed = objective(x_closed) # f(x_closed​)=0.5|​Wxclosed​−b|^2_2
res_closed = torch.linalg.norm(W @ x_closed - b) #res_closed=|Wxclosed​−b|_2​

# fclosed ​= 0.5​⋅(res_closed)2

# ===== 3) Gradient Descent =====
def estimate_sigma_max_sq(A, iters=20):
    y = torch.randn(A.shape[1], device=A.device)
    y = y / (y.norm() + 1e-12) 
    for _ in range(iters):
        y = A.T @ (A @ y)
        y = y / (y.norm() + 1e-12)
    Ay = A @ y
    return float(Ay @ Ay)  # ≈ σ_max(A)^2

L_est = estimate_sigma_max_sq(W, iters=25)
eta = 0.9 / L_est
T = 10000                               # more iterations

x = x0.clone()
x_q = torch.zeros(n, device=device)
for _ in range(T):
    x -= eta * grad(x)
    
    # -- Quantized GD step --
    r = Wx_q(x_q) - b
    g = WT_r_q(r)
    x_q -= eta_q * g
    # -----------------------

# ===== 3.5) Metrics vs closed-form x* only =====
tiny = 1e-12

# Recompute objectives/residuals for each candidate using TRUE W (fairness)
f_gd   = objective(x)
res_gd = torch.linalg.norm(W @ x - b)

f_q    = objective(x_q)
res_q  = torch.linalg.norm(W @ x_q - b)

def report_vs_closed(name, xcand, f_cand, res_cand):
    rel_err     = torch.linalg.norm(xcand - x_closed) / (torch.linalg.norm(x_closed) + tiny)
    rel_obj     = abs(f_cand - f_closed) / (abs(f_closed) + tiny)
    cos_sim     = torch.dot(xcand, x_closed) / ((torch.norm(xcand) + tiny) * (torch.norm(x_closed) + tiny))
    rel_res_gap = (res_cand - res_closed) / (res_closed + tiny)
    print(f"{name:16s}: rel ‖x−x*‖ = {rel_err.item():.2e} , "
          f"cos(x, x*) = {cos_sim.item():.6f} , "
          f"rel residual gap = {rel_res_gap.item():.2e} , "
          f"rel obj diff = {rel_obj.item():.2e}")

print(f"Device: {device}")
print(f"Problem size: W={m}x{n}")
print(f"Stepsize eta = {eta:.3e}, iterations = {T}")
print("------------------------------------------------------")
print(f"Closed-form:        residual = {res_closed.item():.6f} , f(x*)   = {f_closed.item():.6f}")
print(f"Gradient descent:   residual = {res_gd.item():.6f} , f(x_T)  = {f_gd.item():.6f}")
print(f"Quantized GD:       residual = {res_q.item():.6f} , f(x_q)  = {f_q.item():.6f}")
print("------------------------------------------------------")
print("All comparisons are vs closed-form x*")
report_vs_closed("Full-prec GD", x,   f_gd, res_gd)
report_vs_closed("Quantized GD", x_q, f_q,  res_q)


# rel_err = ‖x_candidate − x_closed‖₂ / ‖x_closed‖₂
# cos_sim = (x_candidate · x_closed) / (‖x_candidate‖₂ · ‖x_closed‖₂)
# rel_res_gap = (‖W x_candidate − b‖₂ − ‖W x_closed − b‖₂) / ‖W x_closed − b‖₂
# rel_obj = | f(x_candidate) − f(x_closed) | / f(x_closed)


