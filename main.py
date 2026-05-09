"""
DIFFERENTIAL EVOLUTION OPTIMIZER
================================

INPUT:
    - Search Space: 3D space where each variable ranges from -10 to 10
    - Problem: Find x = [x1, x2, x3] that minimizes f(x) = x1^2 + x2^2 + x3^2
    - Starting Point: 10 random solutions scattered in the search space
    - Example Initial: [2.56, 4.12, -4.52] with fitness = 43.91

OUTPUT:
    - Optimal Solution: x = [x1, x2, x3] close to [0, 0, 0]
    - Minimum Fitness: Value close to 0
    - Example Final: [0.556, 0.374, -0.539] with fitness = 0.74

HOW WE GET THERE:
    1. Start with 10 random candidate solutions
    2. Each generation, for each solution:
       a. MUTATION: Combine 3 other random solutions to create variation
       b. CROSSOVER: Mix the mutation with current solution
       c. SELECTION: Keep the new solution if it's better
    3. Repeat for 20 generations, gradually moving toward [0, 0, 0]
    4. Population converges: 43.91 -> 20.27 -> 9.87 -> 2.78 -> 0.74 -> ~0
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Problem Setup
# ------------------------------

dimensions = 3                  # Number of variables to optimize (x1, x2, x3)
bounds = [(-10, 10)] * dimensions  # Each variable can be between -10 and 10
                                    # Creates: [(-10,10), (-10,10), (-10,10)]

population_size = 10            # Number of candidate solutions we maintain
generations = 20                # Number of evolution cycles
F = 0.8   # Mutation factor: Controls exploration strength (0.5-1.0)
          # Higher F = more aggressive exploration
CR = 0.7  # Crossover probability: 70% chance to take from mutant
          # Higher CR = more mixing between solutions

# ------------------------------
# 2️⃣ Objective Function (Sphere)
# ------------------------------
def objective_function(vector):
    """
    Sphere function: f(x) = sum(x_i^2) = x1^2 + x2^2 + x3^2
    
    INPUT: vector = [x1, x2, x3]
           Example: [2, -3, 1]
    
    OUTPUT: sum of squares
            Example: 2^2 + (-3)^2 + 1^2 = 4 + 9 + 1 = 14
    
    GOAL: Minimize (lower is better)
          Optimal solution is [0, 0, 0] with fitness = 0
    
    WHY THIS FUNCTION?
    - Benchmark test for optimization algorithms
    - Known optimum at [0, 0, 0]
    - Simple convex shape (bowl-shaped)
    - Used in machine learning (least squares), physics (energy), robotics
    """
    return sum(x**2 for x in vector)

# ------------------------------
# 3️⃣ Initialize Population
# ------------------------------
"""
INPUT: Empty (start from scratch)

PROCESS: Generate 10 random solutions
    - Each solution has 3 random numbers between -10 and 10
    - Example: [2.56, 4.12, -4.52]

OUTPUT: Population of 10 diverse candidate solutions
"""
population = [
    [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)]
    for _ in range(population_size)
]
# Explanation: 
# Inner loop: [random number, random number, random number] -> one individual
# Outer loop: Create 10 such individuals

# Evaluate fitness (quality) of each solution
fitness = [objective_function(ind) for ind in population]
# Lower fitness = better solution (closer to [0,0,0])

print("="*70)
print("INITIAL STATE (Random Starting Points)")
print("="*70)
for idx, (ind, fit) in enumerate(zip(population, fitness)):
    print(f"Individual {idx+1}: {ind}, Fitness: {fit:.4f}")
print(f"\nBest Initial Fitness: {min(fitness):.4f}")
print(f"Worst Initial Fitness: {max(fitness):.4f}")

# Store history for visualization
history_best = []  # Track best fitness each generation
history_avg = []   # Track average fitness each generation

# ------------------------------
# 4️⃣ Differential Evolution Loop
# ------------------------------
"""
MAIN ALGORITHM - Evolves population over 20 generations

For each generation, for each individual:
    INPUT: Current population + one target individual
    
    STEP 1 - MUTATION: Create variation
        - Pick 3 random individuals (a, b, c)
        - mutant = a + F * (b - c)
        - Example: a=[1,2,3], b=[4,5,6], c=[2,3,4]
                   mutant = [1,2,3] + 0.8*([4,5,6]-[2,3,4])
                          = [1,2,3] + 0.8*[2,2,2]
                          = [2.6, 3.6, 4.6]
    
    STEP 2 - CROSSOVER: Mix mutant with target
        - For each dimension, flip a coin (70% heads)
        - Heads: take from mutant, Tails: keep from target
        - Example: mutant=[2.6,3.6,4.6], target=[1,2,3]
                   Coins: [heads, tails, heads]
                   trial = [2.6, 2, 4.6]
    
    STEP 3 - SELECTION: Keep only if better
        - If trial_fitness < target_fitness: replace target
        - Else: keep target (never get worse!)
    
    OUTPUT: Improved population (fitness decreases over time)
"""
print("\n" + "="*70)
print("EVOLUTION PROCESS (20 Generations)")
print("="*70)
for gen in range(generations):
    for idx, target in enumerate(population):
        # ===== STEP 1: MUTATION =====
        # Select 3 distinct individuals (not including current target)
        indices = [i for i in range(population_size) if i != idx]
        a, b, c = random.sample(indices, 3)
        
        # Create mutant vector: mutant = a + F * (b - c)
        # This creates variation based on population differences
        mutant = [
            population[a][i] + F * (population[b][i] - population[c][i])
            for i in range(dimensions)
        ]
        # Example with numbers:
        # a=[1,2,3], b=[4,5,6], c=[2,3,4], F=0.8
        # difference = b-c = [2,2,2]
        # mutant = [1,2,3] + 0.8*[2,2,2] = [2.6, 3.6, 4.6]
        
        # Boundary check: keep mutant within [-10, 10]
        mutant = [max(bounds[i][0], min(mutant[i], bounds[i][1])) for i in range(dimensions)]
        # If mutant[i] = 12, clamps to 10
        # If mutant[i] = -15, clamps to -10
        
        # ===== STEP 2: CROSSOVER =====
        # Mix mutant with target (70% from mutant, 30% from target)
        trial = [
            mutant[i] if random.random() < CR else target[i]
            for i in range(dimensions)
        ]
        # Example: random() generates [0.2, 0.8, 0.5]
        #          CR = 0.7
        #          0.2 < 0.7 ✓ -> take mutant[0]
        #          0.8 < 0.7 ✗ -> keep target[1]
        #          0.5 < 0.7 ✓ -> take mutant[2]
        
        # ===== STEP 3: SELECTION =====
        # Keep trial only if it's better (greedy selection)
        trial_fitness = objective_function(trial)
        if trial_fitness <= fitness[idx]:
            population[idx] = trial          # Replace with better solution
            fitness[idx] = trial_fitness
        # Else: keep old solution (fitness never gets worse!)
    
    # Record statistics for this generation
    best_idx = np.argmin(fitness)  # Index of best individual
    best_fit = fitness[best_idx]   # Best fitness value
    avg_fit = np.mean(fitness)     # Average fitness of population
    history_best.append(best_fit)
    history_avg.append(avg_fit)
    
    # Print progress every generation
    print(f"\nGeneration {gen+1:2d}: Best = {best_fit:.4f}, Avg = {avg_fit:.4f}")
    print(f"              Best Individual = {population[best_idx]}")

# ------------------------------
# 5️⃣ Final Result
# ------------------------------
"""
OUTPUT: Best solution found after 20 generations

INPUT (Generation 0):  Random solutions, best fitness ~40-150
PROCESS:               20 generations of mutation, crossover, selection
OUTPUT (Generation 20): Near-optimal solution, fitness close to 0

Example Journey:
    Gen 0:  Best = 43.91  (random start)
    Gen 1:  Best = 20.27  (improvement via mutation/crossover)
    Gen 3:  Best = 9.87   (converging toward optimum)
    Gen 8:  Best = 2.78   (close to optimum)
    Gen 20: Best = 0.74   (very close to [0,0,0])
    
If we ran more generations: 0.74 -> 0.3 -> 0.1 -> 0.01 -> ~0
"""
best_idx = np.argmin(fitness)
best_solution = population[best_idx]
best_score = fitness[best_idx]

print("\n" + "="*70)
print("FINAL RESULT")
print("="*70)
print(f"Optimal Solution: {best_solution}")
print(f"Fitness Value:    {best_score:.4f}")
print(f"\nTarget Solution:  [0.0, 0.0, 0.0]")
print(f"Target Fitness:   0.0000")
print(f"\nDistance to optimum: {np.linalg.norm(best_solution):.4f}")
print("="*70)

# ------------------------------
# 6️⃣ Visualization: Fitness over Generations
# ------------------------------
"""
PLOT EXPLANATION:

Blue Line (Best Fitness):
    - Shows the fitness of the best individual each generation
    - Should decrease sharply at first, then plateau near 0
    - Demonstrates the algorithm finding better solutions
    - Example: 43.91 -> 20.27 -> 9.87 -> 2.78 -> 0.74

Orange Line (Average Fitness):
    - Shows the average fitness of all 10 individuals
    - Decreases more gradually than best fitness
    - Shows the entire population is improving, not just the best
    - Gap between lines = population diversity

What Good Convergence Looks Like:
    - Both lines decrease over time (✓ improving)
    - Best line reaches near 0 (✓ found optimum)
    - Lines get closer together (✓ population converging)
    - No increase in fitness (✓ elitism working)
"""
plt.figure(figsize=(10,6))
plt.plot(range(1, generations+1), history_best, label='Best Fitness', linewidth=2, marker='o')
plt.plot(range(1, generations+1), history_avg, label='Average Fitness', linewidth=2, marker='s')
plt.xlabel("Generation", fontsize=12)
plt.ylabel("Fitness (Lower is Better)", fontsize=12)
plt.title("Differential Evolution Convergence on Sphere Function", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n[SUCCESS] Optimization Complete! Check the plot to see convergence.")