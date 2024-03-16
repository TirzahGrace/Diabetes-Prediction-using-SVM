import numpy as np

class SVM:
    def __init__(self, C=0.1, learning_rate=0.001, tolerance=1e-7, max_iterations=5000):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.w = None
        self.b = 1
        self.tolerance = tolerance

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.w = np.ones(n_features)

        iter = 0
        prev_obj_value = float('inf')

        while True:
            # Compute gradients
            grad_w, grad_b = self._compute_gradients(X, y)

            # Update parameters
            self.w -= self.learning_rate * grad_w
            self.b -= 0.01 * self.learning_rate * grad_b      

            obj_value = self._objective_function(X, y)

            # Print progress
            if iter % 100 == 0:
                print(f"Iteration {iter}: Objective function = {obj_value}")
                pass

            # Check convergence criterion
            if (prev_obj_value - obj_value < self.tolerance) or iter == self.max_iterations:
                break               # Stop iteration if convergence criterion is met

            prev_obj_value = obj_value
            iter += 1


    def _objective_function(self, X, y):
        # Compute objective function value
        margins = np.maximum(0, 1 - y * (np.dot(X, self.w) - self.b))
        loss = np.square(margins)
        regularization_term = np.linalg.norm(self.w, ord=1)
        return regularization_term + (self.C / 2) * np.sum(loss)

    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, 0)


    def _compute_gradients(self, X, y):
        n_samples = X.shape[0]
        grad_w = np.zeros(X.shape[1])
        margin = y * (np.dot(X, self.w) - self.b)
        grad_b = 0

        for i in range(n_samples):
            # margin[i] = y[i] * (np.dot(X.values[i], self.w) - self.b)  # Compute margin for each sample
            if margin[i] < 1:
                grad_w += -1 * self.C * y[i] * X.values[i]
                grad_b +=  self.C * y[i]

        # Regularization term for w
        grad_w += np.sign(self.w)

        return grad_w, grad_b

