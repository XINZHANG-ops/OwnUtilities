import json
from functools import partial
from skopt import space, gp_minimize
import time

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def MultiJacobian(f, x, h=.0001):
    Jacobian = []
    for i in range(len(x)):
        x[i] = x[i] + h
        f1 = np.array(f(x))
        x[i] = x[i] - 2 * h
        J1 = (f1 - np.array(f(x))) / (2 * h)
        Jacobian.append(J1)
    return np.transpose(Jacobian)


def getLambda(f, x0, solveHJ, Lambrange, n=5, tol=1e-3):
    start = Lambrange[0]
    end = Lambrange[1]
    V = []
    crit = tol + 10
    while True:
        V = []
        s = np.linspace(start, end, n)
        for o in range(n):
            V.append(f(list(x0 - s[o] * solveHJ)))
        minpos = V.index(min(V))
        if crit <= tol:
            break
        if minpos == 0:
            start = s[0]
        else:
            start = s[minpos - 1]
        if minpos == (n - 1):
            end = s[n - 1]
        else:
            end = s[minpos + 1]
        crit = end - start
    return s[minpos]


class BFGS:
    '''
    Find the Optimal solution of a function f: Rn -> R,n>=2. Given a initial x0 to start the iteration.
    Lambda is True of False to activate the line search or not
    The method is not necessarily coverge
    '''
    def __init__(self, f, x0, Lambda=False, *args, tol=1e-8, maxit=1000, Lambdarange=(-1.5, 1.5)):
        self.fct = f
        self.initial = np.array(x0)
        self.dim = len(x0)
        self.H0 = np.identity(len(x0))
        self.Lambda = Lambda
        self.args = args
        self.tol = tol
        self.maxit = maxit
        self.Lambdarange = Lambdarange

    def solution(self, message=True):
        dim = self.dim
        fct = self.fct
        xk = self.initial
        tol = self.tol
        Hk = self.H0
        Lambda = self.Lambda
        Lambdarange = self.Lambdarange
        maxit = self.maxit
        args = self.args
        Allx = [xk]
        i = 1
        while True:
            Jk = MultiJacobian(fct, list(xk))  # Jk n*1 array
            solveHJ = np.matmul(np.linalg.inv(Hk), Jk)  #solveHJ n*1 array
            if Lambda:
                Lamb = getLambda(fct, xk, solveHJ, Lambdarange)
            else:
                Lamb = 1
            xk1 = xk - Lamb * solveHJ  # xk1,xk n*1 array
            Allx.append(xk1.copy())
            crit = np.sqrt(sum((xk1 - xk)**2)) / (1 + np.sqrt(sum(xk**2)))
            if crit <= tol:
                break
            if i > maxit:
                Warning('Max iteration reached')
                break
            yk = MultiJacobian(fct, list(xk1)) - Jk  # yk n*1 array
            zk = xk1 - xk  # zk n*1 array
            N1 = np.matmul(
                np.matmul(np.matmul(Hk, zk.reshape(dim, 1)), np.transpose(zk.reshape(dim, 1))), Hk
            )
            D1 = np.matmul(np.matmul(np.transpose(zk.reshape(dim, 1)), Hk), zk.reshape(dim, 1))
            N2 = np.matmul(yk.reshape(dim, 1), np.transpose(yk.reshape(dim, 1)))
            D2 = np.matmul(np.transpose(yk.reshape(dim, 1)), zk.reshape(dim, 1))
            Hk1 = Hk - (N1 / D1) + (N2 / D2)
            Hk = Hk1
            xk = xk1
            i += 1
        fsolution = fct(xk1)
        if message:
            print('Optimal Solution is: {}'.format(fsolution))
            print('with X value(s) are: {}'.format(xk1))
        else:
            return {'function': fct, 'Allx': Allx}

    def plot3D(self, x1=None, x2=None, smooth=0.2, mult=2, nstep=np.inf, pause=None, **kwargs):
        results = self.solution(message=False)
        fct = results['function']
        Allx = np.array(results['Allx'])
        if (Allx.shape[1] != 2):
            print('Error: Only provide 3D plot for R2 to R function')
            return
        else:
            if x1 == None and x2 == None:
                minx1 = mult * min(Allx[:, 0])
                maxx1 = mult * max(Allx[:, 0])
                minx2 = mult * min(Allx[:, 1])
                maxx2 = mult * max(Allx[:, 1])
            elif x1 != None and x2 == None:
                minx1 = min(x1)
                maxx1 = max(x1)
                minx2 = mult * min(Allx[:, 1])
                maxx2 = mult * max(Allx[:, 1])
            elif x1 == None and x2 != None:
                minx1 = mult * min(Allx[:, 0])
                maxx1 = mult * max(Allx[:, 0])
                minx2 = min(x2)
                maxx2 = max(x2)
            else:
                minx1 = min(x1)
                maxx1 = max(x1)
                minx2 = min(x2)
                maxx2 = max(x2)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x1 = np.arange(minx1, maxx1, smooth)
            x2 = np.arange(minx2, maxx2, smooth)
            X, Y = np.meshgrid(x1, x2)
            zs = np.array(fct([np.ravel(X), np.ravel(Y)]))
            Z = zs.reshape(X.shape)
            ax.plot_surface(
                X,
                Y,
                Z,
                color=kwargs.get('color', 'blue'),
                antialiased=kwargs.get('antialiased', True),
                alpha=kwargs.get('alpha', 0.2),
                cmap=kwargs.get('cmap', None)
            )
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('f(x1,x2)')
            plt.title('Plot of the convergence')
            for i in range(min(nstep, Allx.shape[0])):
                x1 = Allx[i, 0]
                x2 = Allx[i, 1]
                fx = fct([x1, x2])
                ax.scatter(x1, x2, fx, linewidth=5, color='green')
                ax.text(x1, x2, fx, '{0}'.format(i + 1), fontsize=15)
                if pause == 0 or pause == None:
                    continue
                else:
                    plt.pause(pause)
            plt.show()


def BFGS_demo():
    def g(x):
        return x[0]**2 + x[1]**2

    def f(x):
        return -(np.sin(x[0]))**2 - (np.sin(x[1]))**2

    def h(x):
        a = 0.98
        n = 0.85
        y = np.exp(x[0])
        z = np.exp(x[1])
        A = n * (y**a + z**a)**(n / a - 1)
        R = A * (y**a + z**a) - .62 * y - .6 * z
        return -R

    def w(x):
        x1 = x[0]
        x2 = x[1]
        return (1 - x1)**2 + 100 * (x2 - x1**2)**2

    def F(x):
        x1 = x[0]
        y = x[1]
        return np.sin(.5 * x1**2 - .25 * y**2 + 3) * np.cos(2 * x1 + 1 - np.exp(y))

    #print(help(BFGS))
    #
    # BFGS(g,x0=[1,1],Lambda=True).solution()
    # BFGS(h,x0=[1,1],Lambda=True).solution()
    BFGS(w, x0=[-.5, .5], Lambda=False).solution()
    #BFGS(h,x0=[1,1],Lambda=True).plot3D(x1=[-1,2],x2=[-1,2],cmap='jet',antialiased=False,alpha=.3,pause=.1,nstep=5)
    # BFGS(f,x0=[1,1],Lambda=True).plot3D(x1=[0,3],x2=[0,3],cmap='jet',antialiased=False,alpha=.3)
    # BFGS(g,x0=[1,1],Lambda=True).plot3D(x1=[-1,2],x2=[-1,2],cmap='jet',antialiased=False,alpha=.3)
    BFGS(
        w, x0=[-.6, .5], Lambda=True
    ).plot3D(
        x1=[-1, 1.3], x2=[-.1, 1.3], cmap='jet', antialiased=False, alpha=.3, smooth=.1, pause=.1
    )


class nelder_mead:
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)

        return: tuple (best parameter array, best score)
    '''
    def __init__(
        self,
        f,
        x_start,
        step=0.1,
        no_improve_thr=10e-6,
        no_improv_break=10,
        max_iter=0,
        alpha=1.,
        gamma=2.,
        rho=-0.5,
        sigma=0.5
    ):
        self.parameters = (
            f, x_start, step, no_improve_thr, no_improv_break, max_iter, alpha, gamma, rho, sigma
        )

    def solution(self, result=True):
        f, x_start, step, no_improve_thr, no_improv_break, max_iter, alpha, gamma, rho, sigma = self.parameters

        # init
        x_start = [float(x) for x in x_start]
        x_start = np.array(x_start)
        dim = len(x_start)
        prev_best = f(x_start)
        no_improv = 0
        res = [[x_start, prev_best]]
        Allres = []

        for i in range(dim):
            x = x_start.copy()
            x[i] = x[i] + step
            score = f(x)
            res.append([x, score])

        # simplex iter
        iters = 0
        while 1:
            # order
            res.sort(key=lambda x: x[1])
            Allres.append(res.copy())
            best = res[0][1]

            # break after max_iter
            if max_iter and iters >= max_iter:
                if result:
                    print('Optimal solution is: {0}'.format(res[0][1]))
                    print('With X Value(s) are: {0}'.format(res[0][0]))
                    return res[0]
                else:
                    return Allres
            iters += 1

            # break after no_improv_break iterations with no improvement
            #print('...best so far:', best)

            if best < prev_best - no_improve_thr:
                no_improv = 0
                prev_best = best
            else:
                no_improv += 1

            if no_improv >= no_improv_break:
                if result:
                    print('Optimal solution is: {0}'.format(res[0][1]))
                    print('With X Value(s) are: {0}'.format(res[0][0]))
                    return res[0]
                else:
                    return Allres

            # centroid
            x0 = [0.] * dim
            for tup in res[:-1]:
                for i, c in enumerate(tup[0]):
                    x0[i] += c / (len(res) - 1)

            # reflection
            xr = x0 + alpha * (np.array(x0) - res[-1][0])
            rscore = f(xr)
            if res[0][1] <= rscore < res[-2][1]:
                del res[-1]
                res.append([xr, rscore])
                continue

            # expansion
            if rscore < res[0][1]:
                xe = x0 + gamma * (np.array(x0) - res[-1][0])
                escore = f(xe)
                if escore < rscore:
                    del res[-1]
                    res.append([xe, escore])
                    continue
                else:
                    del res[-1]
                    res.append([xr, rscore])
                    continue

            # contraction
            xc = x0 + rho * (np.array(x0) - res[-1][0])
            cscore = f(xc)
            if cscore < res[-1][1]:
                del res[-1]
                res.append([xc, cscore])
                continue

            # reduction
            x1 = res[0][0]
            nres = []
            for tup in res:
                redx = x1 + sigma * (tup[0] - x1)
                score = f(redx)
                nres.append([redx, score])
            res = nres

    def plot3D(
        self,
        x1range=None,
        x2range=None,
        smooth=.2,
        nstep=np.inf,
        text=False,
        pause=None,
        **kwargs
    ):
        f, x_start, step, no_improve_thr, no_improv_break, max_iter, alpha, gamma, rho, sigma = self.parameters
        Allres = self.solution(result=False)
        if (np.array(Allres).shape[1] != 3):
            print('Error: Only provide 3D plot')
            return
        jet = plt.get_cmap('jet')
        colors = iter(jet(np.linspace(0, 1, len(Allres))))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xmax = -np.inf
        xmin = np.inf
        ymax = -np.inf
        ymin = np.inf
        for i in range(len(Allres)):
            x1 = Allres[i][0][0][0]
            y1 = Allres[i][0][0][1]
            x2 = Allres[i][1][0][0]
            y2 = Allres[i][1][0][1]
            x3 = Allres[i][2][0][0]
            y3 = Allres[i][2][0][1]
            xmax = max(x1, x2, x3, xmax)
            xmin = min(x1, x2, x3, xmin)
            ymax = max(y1, y2, y3, ymax)
            ymin = min(y1, y2, y3, ymin)
        if x1range == None and x2range == None:
            pass
        elif x1range != None and x2range == None:
            xmin = min(x1range)
            xmax = max(x1range)
        elif x1range == None and x2range != None:
            ymin = min(x2range)
            ymax = max(x2range)
        else:
            xmin = min(x1range)
            xmax = max(x1range)
            ymin = min(x2range)
            ymax = max(x2range)
        x1 = np.arange(xmin, xmax, smooth)
        x2 = np.arange(ymin, ymax, smooth)
        X, Y = np.meshgrid(x1, x2)
        zs = np.array(f([np.ravel(X), np.ravel(Y)]))
        Z = zs.reshape(X.shape)
        ax.plot_surface(
            X,
            Y,
            Z,
            color=kwargs.get('color', 'blue'),
            antialiased=kwargs.get('antialiased', True),
            alpha=kwargs.get('alpha', 0.2),
            cmap=kwargs.get('cmap', None)
        )
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1,x2)')
        plt.title('Nelder Mead in 3d')
        for i in range(min(nstep, len(Allres))):
            c = next(colors)
            x1 = Allres[i][0][0][0]
            y1 = Allres[i][0][0][1]
            z1 = Allres[i][0][1]
            x2 = Allres[i][1][0][0]
            y2 = Allres[i][1][0][1]
            z2 = Allres[i][1][1]
            x3 = Allres[i][2][0][0]
            y3 = Allres[i][2][0][1]
            z3 = Allres[i][2][1]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=c)
            ax.plot([x2, x3], [y2, y3], [z2, z3], color=c)
            ax.plot([x3, x1], [y3, y1], [z3, z1], color=c)
            if text:
                xt = (x1 + x2 + x3) / 3
                yt = (y1 + y2 + y3) / 3
                zt = (z1 + z2 + z3) / 3
                ax.text(xt, yt, zt, '{0}'.format(i + 1), fontsize=10, color=c)
            if pause == 0 or pause == None:
                continue
            else:
                plt.pause(pause)
        plt.show()


def nelder_mead_demo():
    def d3(x):
        return x[0]**2 + x[1]**2 + x[2]**2

    def g(x):
        return x[0]**2 + x[1]**2

    def f(x):
        return -(np.sin(x[0]))**2 - (np.sin(x[1]))**2

    def h(x):
        a = 0.98
        n = 0.85
        y = np.exp(x[0])
        z = np.exp(x[1])
        A = n * (y**a + z**a)**(n / a - 1)
        R = A * (y**a + z**a) - .62 * y - .6 * z
        return -R

    def w(x):
        x1 = x[0]
        x2 = x[1]
        return (1 - x1)**2 + 100 * (x2 - x1**2)**2

    def F(x):
        x1 = x[0]
        y = x[1]
        return np.sin(.5 * x1**2 - .25 * y**2 + 3) * np.cos(2 * x1 + 1 - np.exp(y))

    #nelder_mead(h,[0,1]).solution()
    nelder_mead(w, [-.5, .5]).solution()

    # nelder_mead(w,[0.8,1]).plot3D(alpha=.3,nstep=100,smooth=.1,cmap='jet',text=True,pause=.1)
    nelder_mead(f, [0.8, 1]).plot3D(alpha=.3, nstep=20, smooth=.1, cmap='jet', text=True, pause=.2)


class call_back_bayesian:
    def __init__(self, save_path=None, save_per_n_steps=10):
        self.save_path = save_path
        self.steps = 0
        self.save_per_n_steps = save_per_n_steps
        self.time_stamps = []
        self.accuracies = []
        self.configs = []
        self.all = []
        self.start_time = time.time()

    def time_stamp_call(self, res):
        self.time_stamps.append(time.time())

    def accuracy_call(self, res):
        self.accuracies.append(float(res['func_vals'][-1]))

    def config_call(self, res):
        self.configs.append([float(i) for i in res['x_iters'][-1]])

    def all_call(self, res):
        self.all.append(list(res.items()))

    def save_steps(self, res):
        self.steps += 1
        if self.steps % self.save_per_n_steps == 0:
            data = {
                'time_stamps': self.time_stamps,
                'accuracies': self.accuracies,
                'configs': self.configs
            }
            if self.save_path is None:
                with open('bayesian_callbacks.json', 'w') as f:
                    json.dump(data, f)
            else:
                with open(self.save_path, 'w') as f:
                    json.dump(data, f)


class Bayesian:
    def __init__(self, objective_callable, param_space):
        self.objective_callable = objective_callable
        self.param_space = []
        self.param_names = []
        for param_config in param_space:
            self.param_names.append(param_config[-1])
            self.param_space.append(
                space.Real(
                    low=param_config[0],
                    high=param_config[1],
                    prior='uniform',
                    name=param_config[-1]
                )
            )

    def bayesian_optimize(self, params, param_names):
        params = dict(zip(param_names, params))
        objective_value = self.objective_callable(**params)
        return objective_value

    def train(self, n_calls=10, verbose=True, save_path=None):
        optimization_function = partial(self.bayesian_optimize, param_names=self.param_names)
        bayesian_callback = call_back_bayesian(save_path=save_path)
        result = gp_minimize(
            optimization_function,
            dimensions=self.param_space,
            n_calls=n_calls,
            n_initial_points=15,
            verbose=verbose,
            callback=[
                bayesian_callback.time_stamp_call, bayesian_callback.accuracy_call,
                bayesian_callback.config_call, bayesian_callback.all_call,
                bayesian_callback.save_steps
            ]
        )
        return result, bayesian_callback


def demo_bayesian():
    def objective_bayesian(**kwargs):
        x_names = sorted(list(kwargs.keys()))
        x = [kwargs[name] for name in x_names]
        return x[0]**2 + x[1]**2 + x[2]**2

    param_space = []

    for i in range(3):
        param_space.append([-100, 100, f'variable name {i}'])

    tuning = Bayesian(objective_callable=objective_bayesian, param_space=param_space)
    result, bayesian_callback = tuning.train(n_calls=100, verbose=True)
