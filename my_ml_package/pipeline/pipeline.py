class Pipeline:

    def __init__(self, steps):

        self.steps = steps

    def fit(self, X, y):

        for name, step in self.steps[:-1]:

            X = step.fit_transform(X)

        final_step_name, final_step = self.steps[-1]

        final_step.fit(X, y)

    def predict(self, X):

        for name, step in self.steps[:-1]:

            X = step.transform(X)

        final_step_name, final_step = self.steps[-1]

        return final_step.predict(X)