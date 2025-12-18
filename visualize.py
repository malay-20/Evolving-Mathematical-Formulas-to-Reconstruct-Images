import matplotlib.pyplot as plt
import numpy as np
import fitness

def visualize_results(target, generated, baselines, history, filename='results.png'):
    plt.figure(figsize=(12, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(target, cmap='gray', vmin=0, vmax=255)
    plt.title("Target")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(generated, cmap='gray', vmin=0, vmax=255)
    m = fitness.get_metrics(target, generated)
    plt.title(f"Evolved\nMSE: {m['MSE']:.1f}")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(target - generated), cmap='hot')
    plt.title("Error Difference")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(baselines['gradient'], cmap='gray', vmin=0, vmax=255)
    m = fitness.get_metrics(target, baselines['gradient'])
    plt.title(f"Gradient\nMSE: {m['MSE']:.1f}")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(baselines['random'], cmap='gray', vmin=0, vmax=255)
    m = fitness.get_metrics(target, baselines['random'])
    plt.title(f"Random Noise\nMSE: {m['MSE']:.1f}")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    if len(history) > 0:
        plt.plot(history, label='Fitness', color='blue')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Convergence History')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.show()