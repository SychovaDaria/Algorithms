import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




data = """
Brno,0,1094,524,510,923,789,637,198,700,671
Dubrovnik,1094,0,541,723,144,599,456,1292,613,602
Karlovac,524,541,0,136,389,208,82,722,222,126
Ljubljana,510,723,136,0,482,155,188,708,190,121
Makarska,923,144,386,482,0,455,281,1081,468,368
Porec,789,599,208,155,455,0,220,877,56,118 
Plitvice Lakes,627,456,82,188,281,220,0,839,233,134
Prague,198,1292,722,708,1081,877,839,0,844,869
Pula,700,612,222,190,468,56,233,844,0,104
Rijeka,671,602,126,121,368,118,134,869,104,0
"""

#  data into a numpy array
lines = data.strip().split("\n")
cities = [line.split(",")[0] for line in lines]  # extract city names
distance_matrix = np.array([line.split(",")[1:] for line in lines], dtype=float)


def initialize_population(size_population, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size_population)]

population = initialize_population(10, len(distance_matrix))
#Fitness
def calculate_fitness(individual):
    return sum([distance_matrix[individual[i], individual[i+1]] for i in range(-1, len(individual)-1)])

fitness_scores = [calculate_fitness(ind) for ind in population]

# turnajova selekce
def tournament_selection(population, fitness_scores, tournament_size=3):
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness_scores = [fitness_scores[i] for i in selected_indices]
    winner_index = selected_indices[selected_fitness_scores.index(min(selected_fitness_scores))]
    return population[winner_index]

# selection of parents
parent1 = tournament_selection(population, fitness_scores)
parent2 = tournament_selection(population, fitness_scores)

# operatr mutace
def mutate(individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

def crossover_OX(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))

    # copy a segment from  parent to the child
    child[start:end] = parent1[start:end]

    # filling the remaining parts from the second parent
    current_position = end
    for gene in parent2:
        if gene not in child:
            if current_position >= size:
                current_position = 0
            child[current_position] = gene
            current_position += 1

    return child  

#PMX Partially mapped crossover
def crossover_PMX(parent1, parent2):
    size = len(parent1)
    child1, child2 = [None]*size, [None]*size
    start, end = sorted(random.sample(range(size), 2))

    # copy a segment from  parent to the child
    child1[start:end], child2[start:end] = parent2[start:end], parent1[start:end]

    for i in range(start, end):
        # filling child1
        if parent1[i] not in child1:
            j = i
            while child1[j] is not None:
                j = parent1.index(parent2[j])
            child1[j] = parent1[i]

        # filling child2
        if parent2[i] not in child2:
            j = i
            while child2[j] is not None:
                j = parent2.index(parent1[j])
            child2[j] = parent2[i]

    # filling remaining parts
    for i in range(size):
        if child1[i] is None:
            child1[i] = parent1[i]
        if child2[i] is None:
            child2[i] = parent2[i]

    return list(child1), list(child2)




# parameters for the genetic algorithm
num_cities = len(distance_matrix)
size_population = 100
num_generations = 1000
mutation_rate = 0.1
tournament_size = 5

# Initialize population
population = initialize_population(size_population, num_cities)

#loop of the ga
best_fitness = float('inf')
best_individual = None
all_routes = []


for generation in range(num_generations):
    fitness_scores = [calculate_fitness(individual) for individual in population]
    current_best_fitness = min(fitness_scores)
    if current_best_fitness < best_fitness:
        best_fitness = current_best_fitness
        best_individual = population[fitness_scores.index(best_fitness)]
    
    new_population = []
    while len(new_population) < size_population:
        # selection of parents
        parent1 = tournament_selection(population, fitness_scores, tournament_size)
        parent2 = tournament_selection(population, fitness_scores, tournament_size)
        while parent1 == parent2:
            parent2 = tournament_selection(population, fitness_scores, tournament_size)
        
        # crossover
        if random.random() < 0.5:
            child1, child2 = crossover_PMX(parent1, parent2)
        else:
            child1 = crossover_OX(parent1, parent2)
            child2 = crossover_OX(parent2, parent1)
        
        # mutation
        if random.random() < mutation_rate:
            child1 = mutate(child1)
        if random.random() < mutation_rate:
            child2 = mutate(child2)
        
        # adding the new ones to the new population, with population size limit
        if len(new_population) < size_population:
            new_population.append(child1)
        if len(new_population) < size_population:
            new_population.append(child2)

    population = new_population
    all_routes.append(best_individual.copy())
    if generation % 100 == 0:
        print(f'gen {generation}, fitness: {best_fitness}')

print(f'best route found: {best_individual} with distance {best_fitness}')

# coordinates of cities
theta = np.linspace(0, 2 * np.pi, len(cities), endpoint=False)
cities_coordinates = np.column_stack((np.cos(theta), np.sin(theta)))



    
    
# animation


fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

def init():
    ax.clear()
    ax.scatter(cities_coordinates[:, 0], cities_coordinates[:, 1], color='blue')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    return ax,

def update(frame):
    ax.clear()
    route = all_routes[frame]
    route += [route[0]]  # start=end
    x, y = zip(*[cities_coordinates[i] for i in route])
    ax.scatter(cities_coordinates[:, 0], cities_coordinates[:, 1], color='blue')
    ax.plot(x, y, 'r-')
    return ax,

ani = FuncAnimation(fig, update, frames=len(all_routes), init_func=init, blit=False, repeat=False)

plt.show()


# plot of best route

plt.figure(figsize=(10, 8))

#cities
for i, coord in enumerate(cities_coordinates):
    plt.scatter(*coord, color='blue')
    plt.text(coord[0], coord[1], ' ' + cities[i], fontsize=9)

# best route
best_route_coords = [cities_coordinates[i] for i in best_individual + [best_individual[0]]]
x, y = zip(*best_route_coords)
plt.plot(x, y, 'r-', marker='o', markerfacecolor='green', markersize=10, linewidth=2)

# start
start_x, start_y = cities_coordinates[best_individual[0]]
plt.scatter([start_x], [start_y], c='gold', s=150, label='Start/Finish', edgecolors='black', zorder=5)


plt.legend([f'Best route (fitness: {best_fitness:.2f})'])


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Best route')
plt.axis('equal')
plt.show()



