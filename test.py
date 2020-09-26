
class HopfieldNetwork:
    
    def __init__(self, bias=None):
        self.weights = None
        self.bias = bias
        self.positions_to_update = []

    # Define a function for computing the weights matrix
    def train(self, patterns, synchronous=False, average_activity=None, check=True,warn=True):
        if average_activity is not None:
            mu = average_activity
            self.weights = np.sum([(p.reshape((-1,1))-mu)@(p.reshape((1,-1))-mu) for p in patterns], 
                                  axis=0)
        else:
            self.weights = np.sum([p.reshape((-1,1))@p.reshape((1,-1)) for p in patterns], axis=0)
        self.weights = self.weights.astype(np.float64).copy()
        self.weights /= self.weights.shape[0]
        
        # Check if all patterns are well stored by recalling (if required)
        if check:
            successes = 0
            for pattern in patterns:
                recalled = self.recall(pattern, synchronous=synchronous)
                if np.array_equal(recalled, pattern):
                    successes += 1
                elif warn:
                    print("Warning: input pattern is not a fixed point ", pattern, recalled)
            if warn:
                print(successes, " patterns memorized successfully")
            
            return successes
    
    # Define a function for doing 1 recall (updating once all pattern dimensions)
    def update(self, pattern, synchronous=False, seed=0):
        if synchronous:
            if self.bias is not None:
                pattern_update = np.sign(self.weights@pattern-self.bias)
                pattern_update[pattern_update==0] = 1
                pattern_update = 0.5+0.5*pattern_update
            else:
                pattern_update = np.sign(self.weights@pattern)
                pattern_update[pattern_update==0] = 1
        else:
            # List all patterns dimensions not updated yet (in this complete recall)
            if not self.positions_to_update:
                self.positions_to_update = [i for i in range(pattern.shape[0])]
            
            # Select randomly one position to update
            np.random.seed(seed)
            j = np.random.choice(self.positions_to_update)
            
            # Update the chosen position
            pattern_update = pattern.copy()
            if self.bias is not None:
                value = np.sign(np.sum(self.weights[j,:]*pattern)-self.bias)
                pattern_update[j] = 0.5+0.5*(1 if value>=0 else -1)
            else:
                value = np.sign(np.sum(self.weights[j,:]*pattern))
                pattern_update[j] = 1 if value>=0 else -1
            self.positions_to_update.remove(j)
        
        return pattern_update
    
    # Define a function for doing a complete recall (all pattern dimensions)
    def recall(self, pattern, synchronous=False, max_iterations=None, return_iterations=False):
        iteration = 0
        while not(max_iterations is not None and iteration<max_iterations):
            if synchronous:
                pattern_update = self.update(pattern, synchronous=True)
            else:
                pattern_update = pattern
                for _ in range(pattern.shape[0]):
                    pattern_update = self.update(pattern_update, synchronous=False)
            if np.array_equal(pattern_update, pattern):
                break
            else:
                iteration += 1
                pattern = pattern_update
        
        if return_iterations:
            return pattern, iteration
        else:
            return pattern