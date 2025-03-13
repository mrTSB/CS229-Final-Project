import math
from collections import deque
from torch.optim import Optimizer

def fractional_deriv_coeffs(alpha, K):
    """
    Compute truncated coefficients for the fractional derivative:
        c_j = (-1)^j * Gamma(alpha+1) / ( Gamma(alpha+1 - j) * j! )
    for j = 0, 1, ..., K-1.
    """
    coeffs = []
    for j in range(K):
        # Gamma(alpha+1)
        num = math.gamma(alpha + 1.0)
        # Gamma(alpha+1 - j) * j!
        denom = math.gamma(alpha + 1.0 - j) * math.factorial(j)
        sign = (-1.0)**j
        c_j = sign * (num / denom)
        coeffs.append(c_j)
    return coeffs

class AdaGL(Optimizer):
    """
    Implementation of the AdaGL update rule from Algorithm 1:
      g_t  <- (1 / (nu^alpha)) * SUM_j [ fractional-derivative-coeff_j * grad_{t-j} ]
      m_t  <- beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t  <- beta2 * v_{t-1} + (1 - beta2) * (g_t^2)
      m_hat <- m_t / (1 - beta1^t)
      v_hat <- v_t / (1 - beta2^t)
      C_t  <- [1 - beta2 + (beta2^t)]^(-1)
      theta_t <- theta_{t-1} - lr * C_t * m_hat / sqrt(v_hat + eps)

    The fractional derivative g_t is approximated by storing the last K gradients.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        alpha=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        K=5,
        weight_decay=0.0
    ):
        """
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate (eta).
            alpha (float): Fractional order alpha in the derivative.
            betas (Tuple[float, float]): Coefficients used for running averages (beta1, beta2).
            eps (float): Term added to the denominator for numerical stability.
            K (int): Number of past gradients to store for the fractional derivative approximation.
            weight_decay (float): Weight decay (L2 penalty).
        """
        defaults = dict(
            lr=lr,
            alpha=alpha,
            betas=betas,
            eps=eps,
            K=K,
            weight_decay=weight_decay
        )
        super(AdaGL, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            K = group["K"]
            wd = group["weight_decay"]

            # Pre-compute the fractional derivative coefficients for this alpha (shared by all params).
            # You could cache them if alpha/K do not change across steps.
            coeffs = fractional_deriv_coeffs(alpha, K)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Weight decay (L2 regularization) if specified
                if wd != 0:
                    grad.add_(p.data, alpha=wd)

                state = self.state[p]

                # State initialization
                if "step" not in state:
                    state["step"] = 0
                    # First and second moment buffers
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    # Deque to store the last K gradients for fractional derivative
                    state["grad_queue"] = deque([], maxlen=K)

                m, v = state["m"], state["v"]
                grad_queue = state["grad_queue"]

                # Update step
                state["step"] += 1
                t = state["step"]

                # 1) Push the new gradient to the front of the queue
                grad_queue.appendleft(grad.clone())

                # 2) Approximate the fractional derivative g_t
                #    g_t = (1/(nu^alpha)) * SUM_{j=0..K-1} [ c_j * grad_{t-j} ]
                #    We use nu=1 for simplicity => factor = 1.
                g_t = torch.zeros_like(grad)
                for j, past_grad in enumerate(grad_queue):
                    g_t.add_(past_grad, alpha=coeffs[j])

                # 3) Update biased first moment m_t
                #    m_t = beta1*m_{t-1} + (1-beta1)*g_t
                m.mul_(beta1).add_(g_t, alpha=1 - beta1)

                # 4) Update biased second moment v_t
                #    v_t = beta2*v_{t-1} + (1-beta2)*(g_t^2)
                v.mul_(beta2).addcmul_(g_t, g_t, value=(1 - beta2))

                # 5) Compute bias-corrected first/second moments
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # 6) C_t = [1 - beta2 + (beta2^t)]^-1
                C_t = 1.0 / (1.0 - beta2 + (beta2**t))

                # 7) Update parameters:
                #    theta_t = theta_{t-1} - lr * C_t * m_hat / (sqrt(v_hat) + eps)
                denom = v_hat.sqrt().add_(eps)
                p.data.addcdiv_(m_hat, denom, value=(-lr * C_t))

        return loss
