import time


class EvaluationsScoreTracker():
    def __init__(
        self,
        max_evaluations: int,
        stop_at_score: float,
        print_period: int = 10
    ):
        self.max_evaluations = max_evaluations
        self.stop_at_score = stop_at_score
        self.evaluations = 0
        self.score = float("inf")
        self.history = [(0, float("inf"))]
        self._scale_by = 1
        self.print_period = print_period
        self.prev_print_time = None
        self.print_stopped_text = False
        self.n_mappings = {}
        self.runtime = {}
    
    def add_evaluation(self, n_evaluations: int, best_score: float):
        self.evaluations += n_evaluations * self._scale_by
        self.score = min(self.score, best_score)
        # Same score as before, remove the last entry
        if len(self.history) > 2 and self.history[-2][1] == self.score:
            self.history.pop(-1)
        self.history.append((self.evaluations, self.score))
        
        cur_time = time.time()
        if self.prev_print_time is None or cur_time - self.prev_print_time > self.print_period:
            self.prev_print_time = cur_time
            print(f"Evaluations: {self.evaluations}, Score: {self.score}")

        if self.max_evaluations is not None and self.evaluations > self.max_evaluations:
            self.clean_history()
            if not self.print_stopped_text:
                print(f'Stopping due to evaluations {self.evaluations} > {self.max_evaluations}')
                self.print_stopped_text = True
            return True
        if self.stop_at_score is not None and self.score < self.stop_at_score:
            self.clean_history()
            if not self.print_stopped_text:
                print(f'Stopping due to score {self.score} < {self.stop_at_score}')
                self.print_stopped_text = True
            return True
        return False
    
    def multiply_scale_by(self, scale_by: float):
        self._scale_by *= scale_by
        
    def __repr__(self):
        return f"Evaluations: {self.evaluations}, Score: {self.score}"
    
    def __str__(self):
        return f"Evaluations: {self.evaluations}, Score: {self.score}"
    
    def clean_history(self):
        keep_indices = [0]
        for i in range(1, len(self.history) - 1):
            if self.history[i][1] != self.history[i-1][1] or self.history[i][1] != self.history[i+1][1]:
                keep_indices.append(i)
        keep_indices.append(len(self.history)-1)
        self.history = [self.history[i] for i in keep_indices]
    
    def merge_with(self, other: "EvaluationsScoreTracker"):
        self.score = min(self.score, other.score)
        self.evaluations += other.evaluations

        i, j = 1, 1
        history = [(0, float("inf"))]
        cur_score = float("inf")
        cur_evaluations = 0
        while i < len(self.history) or j < len(other.history):
            # Grab whichever has the lowest evaluations
            if i < len(self.history) and  (j == len(other.history) or self.history[i][0] < other.history[j][0]):
                new_evaluations = self.history[i][0] - self.history[i-1][0]
                new_score = self.history[i][1]
                cur_evaluations += new_evaluations
                cur_score = min(cur_score, new_score)
                history.append((cur_evaluations, cur_score))
                i += 1
            elif j < len(other.history):
                new_evaluations = other.history[j][0] - other.history[j-1][0]
                new_score = other.history[j][1]
                cur_evaluations += new_evaluations
                cur_score = min(cur_score, new_score)
                history.append((cur_evaluations, cur_score))
                j += 1
        self.history = history
        self.clean_history()
            
    def increase_all_evaluations(self, n_evaluations: int):            
        self.evaluations += n_evaluations
        self.history = [(e + n_evaluations, s) for e, s in self.history]
