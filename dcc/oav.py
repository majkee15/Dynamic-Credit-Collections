from dcc.aav import AAV, Parameters
from dcc.base import Base
import numpy as np
from scipy.optimize import fsolve, brentq
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import pickle


class OAV(Base):
    """
    Instance of the controlled Collection process as per Chehrazi, Weber & Glyn 2019
    “Dynamic Credit-Collections Optimization”.
    The value function is constructed in an itterative manner using bivariate interpolation.
    """

    def __init__(self, p, balance, lstart, nx=100, ny=20, lmax=2):
        """
        Standard Class initialization.
        Args:
            p: class Parameters
            balance: double
                starting balance
            lstart: double
                starting intensity
            nx: int or np.ndarray(dim=1)
                number of grid discretizations in balance space
            ny: int or np.ndarray(dim=1)
                number of grid discretizations in lamdba space
            lmax: double
                upper bound of the lambda for discretization.
        """
        super().__init__(__class__.__name__)
        self.lstart = lstart
        self.p = p
        self.balance = balance
        self.aavc = AAV(p)
        self.autonomous_value = self.aavc.u(self.lstart, self.balance)
        self.w_ = self.aavc.w_
        self.w0star = self.aavc.w0star
        self.iw = self.comp_iw()
        self.wistar = self.comp_wistar()

        # Grid specifications
        # The value function is computed in an iterative manner on a given discretized grid:
        self.nx = nx
        self.ny = ny
        if isinstance(nx, int) and isinstance(ny, int):
            self.lambdas_vector = np.linspace(0, lmax, self.ny)
            self.w_vector = np.linspace(0, balance, self.nx)
        elif isinstance(nx, np.ndarray) and isinstance(ny, np.ndarray):
            self.w_vector = nx
            self.lambdas_vector = ny
        self.xx, self.yy = np.meshgrid(self.w_vector, self.lambdas_vector)
        self.zz = np.zeros_like(self.xx)
        self.lambdastars = np.zeros_like(self.w_vector)

        # r integration grid for v bar
        self._r_grid = np.linspace(self.p.r_, 1 - 10e-16, 40)

    def comp_wistar(self):
        # TODO: When instance is reinitialized with new balance this should be auto recomputed.
        wiarray = np.cumsum(np.ones(self.iw))
        wiarray = self.w0star * np.power(1 - self.p.r_, -wiarray)
        wiarray = np.insert(wiarray, 0, self.w0star)
        wiarray = np.insert(wiarray, 0, 0)
        return wiarray

    def comp_iw(self):
        # TODO: When instance is reinitialized with new balance this should be auto recomputed.
        iwret = np.ceil(np.log(self.aavc.w0star / self.balance) / np.log(1 - self.p.r_)).astype(int)
        return iwret

    def lambda0star(self, warr):
        """
        Computes lambda stars, i.e., optimal intervention intensity for terminal value accounts for a given
        grid of balances.
        Args:
            warr: np.ndarray(dim=1)

        Returns: np.ndarray(dim=1)

        """
        if isinstance(warr, float):
            warr = np.array([warr])
        res = np.zeros_like(warr)
        for i, w in enumerate(warr):
            def eqlam(l): return -np.trapz(np.exp(-self.p.rho * self.aavc.t - l * self.aavc.alpha
                                                  - self.p.kappa * self.aavc.beta) * self.aavc.alpha, self.aavc.t) \
                                 * w * self.p.rho + self.p.chat

            res[i] = fsolve(eqlam, np.ones_like(w))
            # res[i] = brentq(eqlam, self.p.lambdainf, 100)
        return res

    def v0star(self, l, w, lstar=None):
        """
        Computes the v0star. For balances smaller than minimal actionable balance this coincides with u(l, w).
        Args:
            l: double
                intensity to be evaluated at
            w: double
                balance to be evaluated at
            lstar: double (optional)
                optimal sustain level given by lambdastar0 equation, if not provided function computes it
                on its own.

        Returns: double
            terminal value of a given account
        """

        if w < self.w_:
            v = self.aavc.u(l, w)
        else:
            if lstar is None:
                lstar = self.lambda0star(w)[0]
            if l > lstar:
                v = self.aavc.u(l, w)
            else:
                astar = (lstar - l) / self.p.delta2
                v = self.aavc.u(l + astar * self.p.delta2, w) + astar * self.p.c
        return v

    def q(self, l, lhat):
        """
        Q function from the text.
        Args:
            l: double
            lhat: double

        Returns: double
            q-value
        """
        with np.errstate(all='raise'):
            # there was a problem with numpy exp that does not handle well negative base with real exponents
            try:
                base = np.divide(lhat - self.p.lambdainf, l - self.p.lambdainf)
                exponent = (self.p.rho + self.p.lambdainf) / self.p.kappa
                res = lhat / (self.p.rho + lhat) * \
                      (np.sign(base) * np.abs(base) ** exponent) * np.exp((lhat - l) / self.p.kappa)
            except Exception:
                print('CAUGHT')
                base = np.divide(lhat - self.p.lambdainf, l - self.p.lambdainf)
                exponent = (self.p.rho + self.p.lambdainf) / self.p.kappa
                self.logger.exception(f'Problem in q. q={0}, '
                                      f'base ={base}, '
                                      f'exponent={exponent}.')
                res = 0
        return res

    def evaluate_v(self, vstar, wi):
        """
        Evaluates the value function up to known wistar on a prespecified grid.
        Args:
            vstar: value funnction to be evaluated
            wi: order of wistar

        Returns: np.ndarray(dim=2)
            Value function evaluated on the grid specified in __init__.
        """

        for j, x in enumerate(self.w_vector):
            if self.wistar[wi] < x <= self.wistar[wi+1]:
                lambdastar = 0.0
                if wi >= 1:
                    lambdastar = self.fi(vstar, x)
                if wi == 0:
                    if x > self.w_:
                        lambdastar = self.lambda0star(x)[0]
                self.lambdastars[j] = lambdastar
                for i, y in enumerate(self.lambdas_vector):
                    if wi >= 1:
                        self.zz[i, j] = self.sustain_operator(vstar, y, x, lambdastar)
                    elif wi == 0:
                        self.zz[i, j] = self.v0star(y, x, lambdastar)
                    else:
                        raise IndexError('Something wrong with the iteration')
        return self.zz


    def interpolate_known_v(self):
        interpolator = RectBivariateSpline(self.lambdas_vector, self.w_vector, self.zz)
        return interpolator

    def plot_interpolator(self):
        interpolator = self.interpolate_known_v()
        zz = interpolator(self.lambdas_vector, self.w_vector)
        fig, ax = plt.subplots()
        CS = ax.contour(self.xx, self.yy, zz)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_title('Interpolated Value Function')
        fig.show()

    def sustain_operator(self, v, lam, w, lhat):
        res = 0
        if lam >= lhat:
            def integrand(l):
                return self.vbar(v, l, w) * self.q(lam, l) * \
                       np.divide(l + self.p.rho, self.p.kappa * (l - self.p.lambdainf))

            y, err = integrate.quad(integrand, lhat, lam)
            res = y + (self.vbar(v, lhat, w) + self.p.chat * \
                       self.p.kappa * np.divide(lhat - self.p.lambdainf, lhat)) * self.q(lam, lhat)
        else:
            res = self.p.chat * (lhat - lam) + (self.vbar(v, lhat, w) + self.p.chat * \
                                                self.p.kappa * np.divide(lhat - self.p.lambdainf, lhat)) * self.q(lhat,
                                                                                                                  lhat)
        return res

    def fi(self, v, w):
        """
         Eq. 30 from Chehrazi and Weber
        Args:
            v: last known value function
            w: balance

        Returns: double
            Lambdastar - optimal intensity sustain level for a given value function and balance.
        """
        def eqfi(lambdabar):
            h = 0.001
            central_diff = (self.vbar(v, lambdabar + h, w) - self.vbar(v, lambdabar - h, w)) / (2 * h)
            return self.p.chat * np.divide((lambdabar + self.p.rho) ** 2,
                                           self.p.kappa * (self.p.rho + self.p.lambdainf)) + \
                   self.p.chat + np.divide(self.p.rho, self.p.kappa * (self.p.rho + self.p.lambdainf)) * \
                   self.vbar(v, lambdabar, w) + \
                   np.divide(lambdabar * (lambdabar + self.p.rho), self.p.kappa * (self.p.rho + self.p.lambdainf)) * \
                   central_diff

        # lambdabars = np.linspace(-5, 5, 50)
        # ys = np.zeros_like(lambdabars)
        # for i, lambar in enumerate(lambdabars):
        #     ys[i] = eqfi(lambar)
        # plt.plot(lambdabars, ys)
        # plt.axhline(0)
        # plt.show()
        # lambdastar = fsolve(eqfi, np.array([0.2]))[0]
        lambdastar = brentq(eqfi, self.p.lambdainf, 100)
        try:
            assert lambdastar > 0
        except AssertionError as err:
            self.logger.exception(
                f'The optimal sustain level Lambda star has to be greater than 0. Current value: {lambdastar:.4f} for w: {w:.2f}.')
            raise err
        return lambdastar

    def solve_v(self, plot_progression_flag: bool=False):
        """
        Iteratively solves the value function.
        Args:
            plot_progression_flag: bool
                plots the iterative progregression of the v.f. approximation

        Returns: None

        """
        self.logger.info('Launching the value function procedure.')
        v_current = self.v0star
        for wi, wval in enumerate(self.wistar[:-1]):
            self.logger.info(f'Computing the value function on ({self.wistar[wi]:.2f}, {self.wistar[wi+1]:.2f}].')
            self.evaluate_v(v_current, wi)
            # for plotting the iterative progress of the value function uncomment the following line
            if plot_progression_flag:
                self.plot_vf().show()
            v_current = self.interpolate_known_v()
            # self.plot_interpolator()
            # if wi > 1:
            #     w_test = np.linspace(wval, self.balance, 20)
            #     vals = np.zeros_like(w_test)
            #     for i ,singlew in enumerate(w_test):
            #         vals[i] = self.fi(v_current, singlew)
            #     plt.plot(w_test, vals, marker = 'x')
            #     plt.show()
            # v_current = lambda l, w: self.extended_value_function(l, w, interpolated)
        pass

    def vbar(self, v, l, w):
        def I(r): return self.p.rdist(r) * (v(l + self.p.delta10 + self.p.delta11 * r, (1 - r) * w) - r * w)

        valgrid = np.array([I(r) for r in self._r_grid]).flatten()
        res = integrate.trapz(valgrid, self._r_grid)
        return res

    def plot_statespace(self):
        '''
        Plots the statespace together with the optimal frontier
        Args:
            warr:

        Returns:

        '''
        fig, ax = plt.subplots()
        ax.axhline(y=self.p.lambdainf, linestyle='-.', color='red')
        ax.axvline(x=self.w_, linestyle='--', color='yellow')
        ax.axvline(x=self.w0star, linestyle='--', color='green')
        for wstar in self.wistar[2:]:
            ax.axvline(x=wstar, linestyle='--', color='black')

        h1 = ax.plot(self.w_vector, self.lambdastars)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Balance')
        ax.set_ylabel('Intensity')
        ax.legend([h1[0]], ['Holding region'])
        return fig

    def plot_vf(self, plot_aav_flag=False):
        """
        Contour plot of the value function for the grid setup in __init__
        Args:
            plot_aav_flag: pot also the autonomous account value

        Returns: figure handle
            fig
        """
        fig, ax = plt.subplots()
        cntr1 = ax.contour(self.xx, self.yy, self.zz, colors='C0')
        ax.clabel(cntr1, inline=1, fontsize=10)
        if plot_aav_flag:
            xx_dump, yy_dump, zz_aav = self.aavc.evaluate_aav(self.lambdas_vector, self.w_vector)
            cntr2 = ax.contour(self.xx, self.yy, zz_aav, colors='C1')
            ax.clabel(cntr2, inline=1, fontsize=10)
            h1, _ = cntr1.legend_elements()
            h2, _ = cntr2.legend_elements()
            ax.legend([h1[0], h2[0]], ['Optimal account value', 'Autonomous account value'])
        else:
            h1, _ = cntr1.legend_elements()
            ax.legend([h1[0]], ['Optimal account value'])
        ax.set_title('Value Function Evaluated on a grid.')
        ax.set_ylabel('Intensity')
        ax.set_xlabel('Balance')
        return fig

    def save(self, filename):
        f_name = filename + '.pickle'
        with open(f_name, "wb") as file_:
            pickle.dump(self, file_, -1)

    @classmethod
    def load(cls, f):
        with open(f +'.pickle', 'rb') as pickle_file:
            return pickle.load(pickle_file)


def _main():
    """
    Dedicated for performance profiling
    Returns: None
    """
    w_start = 100
    lstart = 1
    p = Parameters()
    oav = OAV(p, w_start, lstart)
    oav.solve_v()


if __name__ == '__main__':
    # REFERENCE PARAMETERS THAT ARE PICKLED!
    #
    # w_start = 100
    # lstart = 1
    # p = Parameters()
    # w_array = np.linspace(0, 100, 40)
    # l_array = np.linspace(0, 2, 10)
    # oav = OAV(p, w_start, lstart, nx=200, ny=20)
    # oav.solve_v()
    # oav.plot_vf(plot_aav_flag=True)
    # oav.plot_statespace()
    # plt.show()
    # oav.save('ref_parameters')

    # Load pickled file here

    filename = 'dcc/ref_parameters'
    oav = OAV.load(filename)
    oav.plot_statespace()
    oav.plot_vf(plot_aav_flag=True)
    plt.show()



    # Performance profiling
    # import cProfile
    # cProfile.run('_main()')

    print(oav.__doc__)
