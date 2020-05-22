from base.base_evaluator import BaseEvaluator

class TestEvaluator(BaseEvaluator):

    def __init__(self, model, data, config):
        super(TestEvaluator, self).__init__(model, data, config)
        print("Setup a test evaluator.")

    def evaluate(self):
        print("TestEvaluator")
        print(self.model, self.data, self.config)
