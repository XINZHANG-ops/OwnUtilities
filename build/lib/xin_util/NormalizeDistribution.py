import numpy as np
from scipy.optimize import minimize


class VerboseCallback:
    def __init__(self):
        self.counter = 0

    def add_one(self):
        self.counter += 1


class Normalize:
    """
    normalize long tail distribution to a normal distribution as much as we can between 0 and 100
    all values in original distribution should be >= 0
    """
    def __init__(self, target_mean=50, target_std=15):
        self.target_mean = target_mean
        self.target_std = target_std
        self.sigmoid_p = 0.5
        self.inverse_p = 0.5
        self.N = np.random.normal(loc=target_mean, scale=target_std, size=100000)

    @staticmethod
    def inverse_mapping(x, p=0.3):
        """
        This function mapping values between 0 and 1 to -1 and 1,
        this will convert long tail distribution to normal
        @param x:
        @param p:
        @return:
        """
        return 1 + -1 / (0.5 + 0.5 * x**(p))

    @staticmethod
    def modified_sigmoid(x, p=0.1):
        """
        This function mapping values from 0 and infinite to 0 and 1
        @param x:
        @param p:
        @return:
        """
        sig = 2 * (1 / (1 + np.exp(-p * x)) - 0.5)
        return sig

    @staticmethod
    def transform(values, sig_p, inv_p):
        """
        transform original distribution with P values
        @param values:
        @param sig_p:
        @param inv_p:
        @return:
        """
        return -100 * Normalize.inverse_mapping(Normalize.modified_sigmoid(values, sig_p), inv_p)

    @staticmethod
    def inverse_transform(score, sig_p, inv_p):
        """
        get resulting values back to original value with P values
        @param score:
        @param sig_p:
        @param inv_p:
        @return:
        """
        score = np.array(score)
        score = -score / 100
        term1 = (2 / (1 - score) - 1)**(1 / inv_p)
        term2 = 2 / (1 + term1) - 1
        return -np.log(term2) / sig_p

    @staticmethod
    def get_density_values(dist):
        """
        get normal frequencies for optimization
        @param dist:
        @return:
        """
        all_count = 1e-6
        count_dict = dict([(i, 0) for i in range(0, 101)])
        for i in dist:
            if 0 <= i and i <= 100:
                all_count += 1
                count_dict[int(i)] += 1
        return [v / all_count for k, v in sorted(count_dict.items(), key=lambda item: item[0])]

    def run_optmization(self, values, tol=1e-5, maxiter=10000, fast=False):
        values = np.array(values)
        if fast:

            def objective_function(x):
                # step counter
                count_step.add_one()

                sigmoid_p = x[0]
                transform_p = x[1]

                # get raw scores
                scores = Normalize.transform(values, sigmoid_p, transform_p)

                # get std loss
                std_loss = abs(np.std(scores) - self.target_std)

                # get mean loss
                mean_loss = abs(np.mean(scores) - self.target_mean)

                # total loss
                total_loss = mean_loss + std_loss
                print(f'Step: {count_step.counter}, current loss: {total_loss}')
                return total_loss

        else:
            normal_density = Normalize.get_density_values(self.N)

            def objective_function(x):
                # step counter
                count_step.add_one()

                sigmoid_p = x[0]
                transform_p = x[1]

                # get raw scores
                scores = Normalize.transform(values, sigmoid_p, transform_p)

                # get score point frequencies
                score_density = np.array(Normalize.get_density_values(scores))

                # get distance loss between two vectors
                M = score_density - normal_density
                normal_loss = np.sqrt(np.sum(M**2))

                # get std loss
                std_loss = abs(np.std(scores) - self.target_std)

                # get mean loss
                mean_loss = abs(np.mean(scores) - self.target_mean)

                # total loss
                total_loss = mean_loss + std_loss + normal_loss * 1000
                print(f'Step: {count_step.counter}, current loss: {total_loss}')

                return total_loss

        initial_value = np.array([.5, .1])
        bounds = [(0, 1), (0, 1)]

        count_step = VerboseCallback()
        res = minimize(
            objective_function,
            initial_value,
            method='Powell',
            tol=tol,
            options={'maxiter': maxiter},
            bounds=bounds
        )

        self.sigmoid_p = res.x[0]
        self.inverse_p = res.x[1]

        print(f'sigmoid_p: {self.sigmoid_p}, inverse_p: {self.inverse_p}')


def demo():
    import numpy as np
    import seaborn as sns
    mu, sigma = 3., 1.  # mean and standard deviation
    s = np.random.lognormal(mu, sigma, 1000)
    sns.displot(s)

    dist_normalize = Normalize(70, 15)
    dist_normalize.run_optmization(s, tol=1e-5, maxiter=10000, fast=True)
    new_dist = dist_normalize.transform(s, dist_normalize.sigmoid_p, dist_normalize.inverse_p)
    sns.displot(new_dist)

    dist_normalize.run_optmization(s, tol=1e-5, maxiter=10000, fast=False)
    new_dist = dist_normalize.transform(s, dist_normalize.sigmoid_p, dist_normalize.inverse_p)
    sns.displot(new_dist)

    trans_back = dist_normalize.inverse_transform(
        new_dist, dist_normalize.sigmoid_p, dist_normalize.inverse_p
    )
    sns.displot(trans_back)
