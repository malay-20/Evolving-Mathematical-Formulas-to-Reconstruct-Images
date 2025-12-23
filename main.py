import numpy as np
import random
from ga import GeneticProgramming
from visualize import visualize_results
import fitness
from PIL import Image
import os 

IMAGE_SIZE = (50, 50)
POP_SIZE = 350
GENERATIONS = 500
SAMPLE_SIZE = 400
BASE_MUTATION = 0.25

def load_target_image(filename):
    folder = "images"
    filepath = os.path.join(folder, filename)
    try:
        img = Image.open(filepath).convert('L').resize(IMAGE_SIZE)
        return np.array(img, dtype=float)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.tile(np.linspace(0, 255, 50), (50, 1))

def save_best(formula, score, ssim_val, gen):
    with open("best_formula.txt", "w") as f:
        f.write(f"Generation: {gen}\n")
        f.write(f"MSE-based Score: {score:.4f}\n")
        f.write(f"SSIM: {ssim_val:.4f}\n") 
        f.write(f"Expression: {formula}\n")

def save_evolution_image(image_array, gen):
    if not os.path.exists('evolution_steps'):
        os.makedirs('evolution_steps')
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(f"evolution_steps/gen_{gen}.png")

def get_pixel_weights(target):
    flat = target.flatten().astype(int)
    counts = np.bincount(flat, minlength=256)
    counts = np.maximum(counts, 1)
    
    total = len(flat)
    weights = total / counts
    
    weight_map = weights[flat].reshape(target.shape)
    return np.log1p(weight_map) + 1.0

def main():
    print("=== Starting Evolutionary Process ===")
    
    filename = '' #add file name (some test imaeges are in the images folder)
    target_img = load_target_image(filename)
    print(f"Target: {filename} {IMAGE_SIZE}")

    gp = GeneticProgramming(target_img)
    
    h, w = target_img.shape
    m_grid, n_grid = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    
    flat_target = target_img.flatten()
    flat_m = m_grid.flatten()
    flat_n = n_grid.flatten()
    
    flat_weights = get_pixel_weights(target_img).flatten()
    
    print(f"Creating {POP_SIZE} random formulas...")
    population = gp.init_population(POP_SIZE, max_depth=10)
    
    curr_mutation_rate = BASE_MUTATION
    best_ever_score = float('-inf')
    
    history = []
    real_history = []
    
    for gen in range(GENERATIONS):
        
        idx = np.random.randint(0, len(flat_target), SAMPLE_SIZE)
        sub_m = flat_m[idx]
        sub_n = flat_n[idx]
        sub_target = flat_target[idx]
        sub_weights = flat_weights[idx]
        
        scores = []
        for ind in population:
            res = ind.evaluate(sub_m, sub_n)
            
            res = np.nan_to_num(res)
            res = np.clip(res, -1e5, 1e5)
            rmin, rmax = res.min(), res.max()
            if rmax > rmin:
                img_out = 255 * (res - rmin) / (rmax - rmin)
            else:
                img_out = np.full_like(res, 128.0)
            
            error = (sub_target - img_out) ** 2
            weighted_mse = np.mean(error * sub_weights)
            
            scores.append(-weighted_mse - (ind.size() * 0.02))
        
        best_idx = np.argmax(scores)
        best_ind = population[best_idx]

        if gen % 100 == 0:
            print(f"Saving evolution image at generation {gen}...")
            step_img = gp.evaluate_full_image(best_ind)
            save_evolution_image(step_img, gen)
        
        if gen % 20 == 0:
            full_img = gp.evaluate_full_image(best_ind)
            real_score = fitness.get_score(target_img, full_img, best_ind.size())
            
            current_metrics = fitness.get_metrics(target_img, full_img)
            current_ssim = current_metrics['SSIM']
            
            history.append(real_score)
            real_history.append(real_score)
            
            print(f"Gen {gen}: Score={real_score:.2f} | SSIM={current_ssim:.4f} | Mutation={curr_mutation_rate:.2f}")
            
            if real_score > best_ever_score:
                best_ever_score = real_score
                save_best(best_ind, real_score, current_ssim, gen)
                print("  >> New best formula")
            
            if len(real_history) > 3:
                diff = abs(real_history[-1] - real_history[-3])
                if diff < 10:
                    curr_mutation_rate = min(0.8, curr_mutation_rate + 0.1)
                    print(f"  >> Stagnation detected. Boosting mutation!")
                else:
                    curr_mutation_rate = max(0.1, curr_mutation_rate - 0.05)
        else:
            history.append(scores[best_idx])
            
        next_gen = []
        
        ranked_indices = np.argsort(scores)[::-1]
        for i in range(3):
            next_gen.append(population[ranked_indices[i]].copy())
            
        while len(next_gen) < POP_SIZE:
            p1 = gp.tournament(population, scores)
            p2 = gp.tournament(population, scores)
            
            if random.random() < 0.85:
                c1, c2 = gp.crossover(p1, p2)
            else:
                c1, c2 = p1, p2
                
            if random.random() < curr_mutation_rate: c1 = gp.mutate(c1)
            if random.random() < curr_mutation_rate: c2 = gp.mutate(c2)
            
            next_gen.extend([c1, c2])
            
        population = next_gen[:POP_SIZE]

    print("\nEvolution finished. ")
    
    final_scores = [fitness.get_score(target_img, gp.evaluate_full_image(i), i.size()) for i in population]
    best_guy = population[np.argmax(final_scores)]
    final_image = gp.evaluate_full_image(best_guy)
    metrics = fitness.get_metrics(target_img, final_image)
    print(f"Final MSE: {metrics['MSE']:.4f}")
    print(f"Final SSIM: {metrics['SSIM']:.4f}")

    random_formulas = gp.init_population(10, max_depth=5) 
    random_outputs = []
    
    for f in random_formulas:
        random_outputs.append(gp.evaluate_full_image(f))
    
    avg_random_baseline = np.mean(random_outputs, axis=0)

    baselines = {
        'gradient': (gp.m_grid + 1) * 127.5,
        'random': avg_random_baseline
    }
    
    visualize_results(target_img, final_image, baselines, history)

if __name__ == "__main__":
    main()