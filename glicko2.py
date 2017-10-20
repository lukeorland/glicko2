# -*- coding: utf-8 -*-
"""
    glicko2
    ~~~~~~~

    The Glicko2 rating system.

    :copyright: (c) 2012 by Heungsub Lee
    :license: BSD, see LICENSE for more details.
"""
import math


__version__ = '0.0.dev'


#: The actual score for win
WIN = 1.
#: The actual score for draw
DRAW = 0.5
#: The actual score for loss
LOSS = 0.


DEFAULT_RATING = 1500
DEFAULT_DEVIATION = 350
DEFAULT_VOLATILITY = 0.06
TAU = 1.0  # system constant
EPSILON = 0.000001  # convergence tolerance
#: A constant which is used to standardize the logistic function to
#: `1/(1+exp(-x))` from `1/(1+10^(-r/400))`
Q = math.log(10) / 400
SCALE_FACTOR = 173.7178


class Rating(object):

    def __init__(self, rating=DEFAULT_RATING, deviation=DEFAULT_DEVIATION, volatility=DEFAULT_VOLATILITY):
        self.rating = rating
        self.deviation = deviation
        self.volatility = volatility

    def __repr__(self):
        c = type(self)
        args = (c.__module__, c.__name__, self.rating, self.deviation, self.volatility)
        return '%s.%s(rating=%.3f, deviation=%.3f, volatility=%.3f)' % args


class Glicko2(object):

    def __init__(self, rating=DEFAULT_RATING, deviation=DEFAULT_DEVIATION, volatility=DEFAULT_VOLATILITY, tau=TAU, epsilon=EPSILON):
        self.rating = rating
        self.deviation = deviation
        self.volatility = volatility
        self.tau = tau
        self.epsilon = epsilon

    def create_rating(self, rating=None, deviation=None, volatility=None):
        if rating is None:
            rating = self.rating
        if deviation is None:
            deviation = self.deviation
        if volatility is None:
            volatility = self.volatility
        return Rating(rating, deviation, volatility)

    def glicko_to_glicko2(self, rating, ratio=SCALE_FACTOR):
        mu = (rating.rating - self.rating) / ratio
        phi = rating.deviation / ratio
        return self.create_rating(mu, phi, rating.volatility)

    def glicko2_to_glicko(self, rating, ratio=SCALE_FACTOR):
        mu, phi = rating.rating, rating.deviation
        rating = mu * ratio + self.rating
        deviation = phi * ratio
        return self.create_rating(rating, deviation, rating.volatility)

    def reduce_impact(self, rating):
        """The original form is `g(RD)`. This function reduces the impact of
        games as a function of an opponent's RD.
        """
        return 1 / math.sqrt(1 + (3 * rating.deviation ** 2) / (math.pi ** 2))

    def expect_score(self, rating, other_rating, impact):
        return 1. / (1 + math.exp(-impact * (rating.rating - other_rating.rating)))

    def determine_sigma(self, rating, difference, variance):
        """Determines new sigma."""
        phi = rating.deviation
        difference_squared = difference ** 2
        # 1. Let a = ln(s^2), and define f(x)
        alpha = math.log(rating.volatility ** 2)
        def f(x):
            """This function is twice the conditional log-posterior density of
            phi, and is the optimality criterion.
            """
            tmp = phi ** 2 + variance + math.exp(x)
            a = math.exp(x) * (difference_squared - tmp) / (2 * tmp ** 2)
            b = (x - alpha) / (self.tau ** 2)
            return a - b
        # 2. Set the initial values of the iterative algorithm.
        a = alpha
        if difference_squared > phi ** 2 + variance:
            b = math.log(difference_squared - phi ** 2 - variance)
        else:
            k = 1
            while f(alpha - k * math.sqrt(self.tau ** 2)) < 0:
                k += 1
            b = alpha - k * math.sqrt(self.tau ** 2)
        # 3. Let fA = f(A) and f(B) = f(B)
        f_a, f_b = f(a), f(b)
        # 4. While |B-A| > e, carry out the following steps.
        # (a) Let C = A + (A - B)fA / (fB-fA), and let fC = f(C).
        # (b) If fCfB < 0, then set A <- B and fA <- fB; otherwise, just set
        #     fA <- fA/2.
        # (c) Set B <- C and fB <- fC.
        # (d) Stop if |B-A| <= e. Repeat the above three steps otherwise.
        while abs(b - a) > self.epsilon:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b < 0:
                a, f_a = b, f_b
            else:
                f_a /= 2
            b, f_b = c, f_c
        # 5. Once |B-A| <= e, set s' <- e^(A/2)
        return math.exp(1) ** (a / 2)

    def rate(self, rating, series):
        # Step 2. For each player, convert the rating and RD's onto the
        #         Glicko-2 scale.
        rating = self.glicko_to_glicko2(rating)
        # Step 3. Compute the quantity v. This is the estimated variance of the
        #         team's/player's rating based only on game outcomes.
        # Step 4. Compute the quantity difference, the estimated improvement in
        #         rating by comparing the pre-period rating to the performance
        #         rating based only on game outcomes.
        d_square_inv = 0
        variance_inv = 0
        difference = 0
        for actual_score, other_rating in series:
            other_rating = self.glicko_to_glicko2(other_rating)
            impact = self.reduce_impact(other_rating)
            expected_score = self.expect_score(rating, other_rating, impact)
            variance_inv += impact ** 2 * expected_score * (1 - expected_score)
            difference += impact * (actual_score - expected_score)
            d_square_inv += (
                expected_score * (1 - expected_score) *
                (Q ** 2) * (impact ** 2))
        difference /= variance_inv
        variance = 1. / variance_inv
        denom = rating.deviation ** -2 + d_square_inv
        mu = rating.rating + Q / denom * (difference / variance_inv)
        phi = math.sqrt(1 / denom)
        # Step 5. Determine the new value, Sigma', ot the sigma. This
        #         computation requires iteration.
        sigma = self.determine_sigma(rating, difference, variance)
        # Step 6. Update the rating deviation to the new pre-rating period
        #         value, Phi*.
        phi_star = math.sqrt(phi ** 2 + sigma ** 2)
        # Step 7. Update the rating and RD to the new values, Mu' and Phi'.
        phi = 1 / math.sqrt(1 / phi_star ** 2 + 1 / variance)
        mu = rating.rating + phi ** 2 * (difference / variance)
        # Step 8. Convert ratings and RD's back to original scale.
        return self.glicko2_to_glicko(self.create_rating(mu, phi, sigma))

    def rate_1vs1(self, rating1, rating2, drawn=False):
        return (self.rate(rating1, [(DRAW if drawn else WIN, rating2)]),
                self.rate(rating2, [(DRAW if drawn else LOSS, rating1)]))

    def quality_1vs1(self, rating1, rating2):
        expected_score1 = self.expect_score(rating1, rating2, self.reduce_impact(rating1))
        expected_score2 = self.expect_score(rating2, rating1, self.reduce_impact(rating2))
        expected_score = (expected_score1 + expected_score2) / 2
        return 2 * (0.5 - abs(0.5 - expected_score))
