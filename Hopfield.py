import matplotlib.pyplot as plt
import numpy as np


class Hopfield:
    
    def __init__(self):
        self.weights = None
        self.positions_to_update = []

    # Compute the weights matrix
    def train(self, patterns):    
        # Matrix multiply pattern as two column vectors, then sum over all patterns
        self.weights = np.sum([p.reshape((-1,1))@p.reshape((1,-1)) for p in patterns], axis=0)
        np.fill_diagonal(self.weights, 0)
        
        self.weights = self.weights.astype(np.float64).copy()
        self.weights /= self.weights.shape[0]
        
        #print(self.weights)
        
        
    # Check if patterns are sucessfully memorised    
    def check(self, patterns, synchronous=True):        
        successes = 0
        for pattern in patterns:
            recalled = self.recall(pattern, synchronous=synchronous)
            
            if np.array_equal(recalled, pattern):
                successes += 1                
            else:
                print("Warning: input pattern is not a fixed point ", pattern, recalled)

        print(successes, " patterns memorized successfully")
            
        
    

    def update(self, pattern, synchronous=False):       
        if synchronous:
            pattern_update = np.sign(self.weights@pattern)
            # make sure sign(0) = 1
            pattern_update[pattern_update==0] = 1
           
        else:
            # List all patterns dimensions not updated yet (in this complete recall)
            if not self.positions_to_update:
                self.positions_to_update = [i for i in range(pattern.shape[0])]
            
            # Select randomly one position to update
            np.random.seed(2020)
            j = np.random.choice(self.positions_to_update)
            
            # Update the chosen position
            pattern_update = pattern.copy()
            value = np.sign(np.sum(self.weights[j,:]*pattern))
            pattern_update[j] = 1 if value>=0 else -1
            self.positions_to_update.remove(j)
        
        return pattern_update
    
    
    
    def recall(self, pattern, synchronous=False, max_iterations=1):
        iteration = 0
        while(iteration < max_iterations):
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
    
        return pattern
        
   
    # Define a function to find all possible attractors
    def attractors(self, synchronous=True):
        attractors = set()
        all_possible_patterns = binary_patterns(self.weights.shape[1])
        for pattern in all_possible_patterns:
            attractors.add(tuple(self.recall(pattern, synchronous=synchronous, max_iterations=100)))
        
        return np.array([np.array(attractor) for attractor in attractors])
    
    
    def energy(self, state):
        alpha = state.reshape((-1,1))@state.reshape((1,-1))
        #return -np.sum(np.sum(alpha*self.weights,  axis=1), axis=0)
        return -np.sum(alpha*self.weights)
    

def binary_patterns(length):
    for combination in range(2**length):
        # https://stackoverflow.com/questions/25589729/binary-numbers-of-n-digits
        combination_string = bin(combination)[2:].zfill(length)
        yield np.array([1 if x=='1' else -1 for x in combination_string], dtype=np.int8)
      
        
# Check if converges to stored patterns when distorted inputs are given   
def test_recall_distorted(model, pattern_distorted, pattern_expected, synchronous=True, print_result=True, check=1):
    pattern_recalled = model.recall(pattern_distorted, synchronous=synchronous, max_iterations=100)
    good_recall_or_not = np.array_equal(pattern_recalled, pattern_expected)
    
    if print_result:
        if check == 1:
            message = 'Success' if good_recall_or_not else 'Failure'
            print("Input:", pattern_distorted, ", Output:", pattern_recalled, " ", message) 
            
        else:
            print("Input:", pattern_distorted, ", Output:", pattern_recalled)      

    
        
          
#################################################################################### 
# ------------------------------ 2.2 ------------------------------       
# Data to be memorised 
x1 = [-1,-1,1,-1,1,-1,-1,1]
x2 = [-1,-1,-1,-1,-1,1,-1,-1]
x3 = [-1,1,1,-1,-1,1,-1,1]   
# Input patterns         
Training_patterns = np.array([x1, x2, x3])

model = Hopfield()
model.train(Training_patterns)
model.check(Training_patterns)


# ------------------------------ 3.1 ------------------------------
x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1])
x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1])
x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1])

print("Synchronous")
test_recall_distorted(model, x1d, x1, synchronous=True)
test_recall_distorted(model, x2d, x2, synchronous=True)
test_recall_distorted(model, x3d, x3, synchronous=True)

#print("\nAsynchronous")
#test_recall_distorted(model, x1d, x1, synchronous=False)
#test_recall_distorted(model, x2d, x2, synchronous=False)
#test_recall_distorted(model, x3d, x3, synchronous=False)


attractors = model.attractors(synchronous=True)
print("Found {} attractors:".format(len(attractors)))
for attractor in attractors:
      print(np.array(attractor))
# For Synchronous update, I got 64 attractors.      
# For Asynchronous update, the attractors are x1, x2, x3 and their inverses (1 and -1 exchanged)    
# If I dont set the diagonal of the Weight Matrix to 0, I get 14 attractors.
      
      
     
# Dissimilar patterns
xD4 = np.array([1,1,-1,1,-1,-1,1,1])
xD5 = np.array([1,1,1,1,1,-1,1,-1])
xD6 = np.array([1,-1,1,1,1,-1,1,-1])

print("Synchronous")
test_recall_distorted(model, xD4, x1, synchronous=True, check=0)
test_recall_distorted(model, xD5, x2, synchronous=True, check=0)
test_recall_distorted(model, xD6, x3, synchronous=True, check=0)

#print("\nAsynchronous")
#test_recall_distorted(model, xD4, x1, synchronous=False, check=0)
#test_recall_distorted(model, xD5, x2, synchronous=False, check=0)
#test_recall_distorted(model, xD6, x3, synchronous=False, check=0)



# ------------------------------ 3.2 ------------------------------
### Load images and check if they are memorised ###
pict = np.genfromtxt('pict.dat', delimiter=',', dtype=np.int8).reshape(-1,1024)
fig = plt.figure(figsize=(10,10))


#for i, pattern in enumerate(pict[:9,:]):
#    fig.add_subplot(330+i+1)
#    plt.imshow(pattern.reshape(32, 32), cmap='gray')
#    plt.title("Pattern "+ str(i+1))

    
model_pict = Hopfield()
model_pict.train(pict[:3,:])
model_pict.check(pict[:3,:], synchronous=True)
#model_pict.check(pict[:3,:], synchronous=False)


### Plotting the deformed figures ###
fig = plt.figure(figsize=(10,10))

fig.add_subplot(221)
plt.imshow(pict[10-1,:].reshape(32, 32), cmap='gray')
plt.title("Pattern 9")

fig.add_subplot(222)
patterns_recalled = model_pict.recall(pict[10-1,:], synchronous=True, max_iterations=100)
plt.imshow(patterns_recalled.reshape(32, 32), cmap='gray')
plt.title("Pattern 9, Synchronous Recalls")

fig.add_subplot(223)
plt.imshow(pict[11-1,:].reshape(32, 32), cmap='gray')
plt.title("Pattern 10")

fig.add_subplot(224)
patterns_recalled = model_pict.recall(pict[11-1,:], synchronous=True, max_iterations=100)
plt.imshow(patterns_recalled.reshape(32, 32), cmap='gray')
plt.title("Pattern 10, Synchronous Recalls")


### Sequential updates on images ###
def sequential_series(pattern, iterations):
    fig = plt.figure(figsize=(15,10))
    current_iteration = 0
    for i, iteration in enumerate(iterations):
        fig.add_subplot(1, len(iterations), i+1)
        pattern_update = pattern
        for _ in range(current_iteration, iteration):
            pattern_update = model_pict.update(pattern_update, synchronous=False)
        pattern = pattern_update
        plt.imshow(pattern.reshape(32, 32), cmap='gray')
        plt.title("{} Iterations".format(iteration))
        current_iteration = iteration
        
        
sequential_series(pict[9,:], iterations=range(0, 1025, 256))


# ------------------------------ 3.3 ------------------------------
for attractor in model.attractors(synchronous=False):
    print("{} => {}".format(attractor, model.energy(attractor)))
   
pict = np.genfromtxt('pict.dat', delimiter=',', dtype=np.int8).reshape(-1,1024)
model_pict = Hopfield()
model_pict.train(pict[:3,:])

   
fig = plt.figure(figsize=(15,5))
for i, attractor in enumerate(pict[:3,:]):
    fig.add_subplot(1, 3, i+1)
    plt.imshow(attractor.reshape(32, 32), cmap='gray')
    plt.title("Energy is {:1.3e}".format(model_pict.energy(attractor)))
