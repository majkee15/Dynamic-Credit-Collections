from scipy import integrate
from scipy.optimize import fsolve
import numpy as np
from dcc.base import Base
import matplotlib.pyplot as plt


# integrand = @(r) (1-(1-r).*exp(-(p.delta10+p.delta11.*r).*y(1))).*p.rdist(r);
# integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)


class AAV(Base):
    """
    Instance of the autonomous (i.e., not controleld) collection process as per Chehrazi, Weber & Glyn 2019
    “Dynamic Credit-Collections Optimization”.
    """

    def __init__(self, parameters, n_grid_points=201):
        """
        Args:
            parameters: class Parameters
            n_grid_points: int
                number of discretizations for the ODE
        """
        super().__init__(__class__.__name__)
        self.parameters = parameters
        self.alpha = None
        self.beta = None
        self.t = None
        self.w_ = None
        self.w0star = None
        self.logger.info(f'Instantiated @ {self.__class__.__name__}')
        # Since the AAV is dependent on the solution of ODE1 that is independent of the balance
        # it is feasible to precompute it here. Discretization done up to e-7
        self.t = np.linspace(0, 20 / self.parameters.kappa, 201)
        y0 = [0, 0]
        y = np.array(integrate.odeint(self.aav_ode, y0, self.t))
        self.alpha = y[:, 0]
        self.beta = y[:, 1]
        self.w_ = self.compute_w_()
        self.w0star = self.compute_w0star()

    def aav_ode(self, y, t):
        def integrand(r): return (1 - (1 - r) * np.exp(-(self.parameters.delta10 + self.parameters.delta11 * r) * y[0])
                                  ) * self.parameters.rdist(r)

        q = integrate.quad(integrand, 0.1, 1)
        dy1 = -self.parameters.kappa * y[0] + q[0]
        dy2 = self.parameters.lambdainf * y[0]
        dydt = [dy1, dy2]
        return dydt

    def u(self, l, w):
        """
        Calculates the autonomous account value of the account.
        Args:
            l: double
                intensity
            w: double
                balance
        Returns: double
            autonomous acc value
        """

        I = np.exp(-self.parameters.rho * self.t - l * self.alpha - self.parameters.kappa * self.beta)
        q = integrate.trapz(I, self.t)
        av = -(1 - self.parameters.rho * q) * w
        return av

    def compute_w_(self):
        """
        Calculares minimum actionable balance for a given acc.
        Returns: double
            Minimum actionable balance underscore{w}
        """
        I = self.alpha * np.exp(-self.parameters.rho * self.t - self.parameters.kappa * self.beta)
        w = (1 / integrate.trapz(I, self.t)) * (self.parameters.c / self.parameters.delta2) * (1 / self.parameters.rho)
        return w

    def compute_w0star(self):
        """
        Calculates w0 star, i.e., balance for the intersection of \lambda_\infty and w0star.
        Calculaates
        Returns: double
            w0star
        """
        I = self.alpha * np.exp(-self.parameters.rho * self.t - self.parameters.lambdainf * self.alpha -
                                self.parameters.kappa * self.beta)

        def w0stareq(w0star): return -integrate.trapz(I, self.t) * w0star * self.parameters.rho + \
                                     self.parameters.c / self.parameters.delta2

        return fsolve(w0stareq, np.array([50]))[0]

    def evaluate_aav(self, l_array, w_array, plot_flag=False):
        """
        Evaluates AAV for different balances, or on a lambda/w grid for contour plots.
        Creates the aav plot w.r.t. different balance values
        Args:
            l_array: ndarray(dim = 1) or ndarray(dim=2)
            w_array: ndarray(dim = 1) or ndarray(dim=2)
            plot_flag(optional): bool

        Returns:

        """
        if np.isscalar(l_array) and w_array.ndim == 1:
            u_vals = np.zeros_like(w_array)
            for i, w in enumerate(w_array):
                u_vals[i] = self.u(l_array, w)

            if plot_flag:
                fig, ax = plt.subplots()
                ax.plot(w_array, u_vals, marker='o')
                ax.set_ylabel('Account Value')
                ax.set_xlabel('Balance')
                ax.set_title('Autonomous account value function')
                plt.show()
            return u_vals
        elif np.isscalar(w_array) and l_array.ndim == 1:
            u_vals = np.zeros_like(l_array)
            for i, l in enumerate(l_array):
                u_vals[i] = self.u(l, w_array)

            if plot_flag:
                fig, ax = plt.subplots()
                ax.plot(l_array, u_vals, marker='o')
                ax.set_ylabel('Account Value')
                ax.set_xlabel('Intensity')
                ax.set_title('Autonomous account value function')
                plt.show()
            return u_vals
        elif w_array.ndim == 1 and l_array.ndim == 1:
            xx, yy = np.meshgrid(w_array, l_array)
            zz = np.zeros_like(xx)
            for j, x in enumerate(w_array):
                for i, y in enumerate(l_array):
                    zz[i, j] = self.u(y, x)
            if plot_flag:
                fig, ax = plt.subplots()
                CS = ax.contour(xx, yy, zz)
                ax.clabel(CS, inline=1, fontsize=10)
                ax.set_title('Autonomous account value function')
                ax.set_ylabel('Account Value')
                ax.set_xlabel('Balance')
                fig.show()
            return xx, yy, zz
        else:
            self.logger.warning('Something wrong with the dimension of the input. Use either 1-D arrays for')


class Parameters:
    """
    Defines parameters for the controlled Hawkes process.
    """

    def __init__(self):
        self.lamdbda0 = 0.11
        self.lambdainf = 0.1
        self.kappa = 0.7
        self.delta10 = 0.02
        self.delta11 = 0.5
        self.delta2 = 1
        self.rho = 0.06
        self.c = 6
        self.r_ = 0.1
        self.chat = self.c / self.delta2

    def rdist(self, r):
        """
        PDF of the relative distribution
        Args:
            r: double
        Returns: double
        """
        return 1 / (1 - self.r_)


if __name__ == '__main__':
    balance = 75
    params = Parameters()
    aavcl = AAV(params)
    print(aavcl.compute_w0star())
    w_array = np.linspace(0, 100, 40)
    l_array = np.linspace(0, 2, 10)
    aavcl.evaluate_aav(1, w_array, True)
    aavcl.evaluate_aav(l_array, w_array, True)
    aavcl.evaluate_aav(l_array, balance, True)
