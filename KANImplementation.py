import argparse
from kan.experiments import run_pde_experiment
from kan.experiments import run_function_fitting_experiment, plot_function_fitting_results

def main():
    parser = argparse.ArgumentParser(description='Run KAN experiments')
    parser.add_argument('--experiment', type=str, choices=['pde', 'function'], 
                      required=True, help='Type of experiment to run')
    parser.add_argument('--function-type', type=str, choices=['special', 'feynman'],
                      help='Type of function for function fitting experiment')
    parser.add_argument('--function-name', type=str,
                      help='Name of function to fit')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run on (cpu or cuda)')
    
    args = parser.parse_args()

    if args.experiment == 'pde':
        # Run PDE experiment with minimal parameters
        run_pde_experiment(
            grid_sizes=[3, 5],  # Very few grid sizes
            kan_shapes=[(2, 5, 1)],  # Single shape
            mlp_depths=[2],  # Single depth
            mlp_width=20,  # Small width
            n_epochs=50,  # Fewer epochs
            n_interior=1000,  # Reduced interior points
            n_boundary=100,  # Reduced boundary points
            device=args.device
        )
    else:
        if not args.function_type or not args.function_name:
            raise ValueError("Must specify --function-type and --function-name for function fitting experiment")
            
        results = run_function_fitting_experiment(
            function_type=args.function_type,
            function_name=args.function_name
        )
        plot_function_fitting_results(results, args.function_name)

if __name__ == '__main__':
    main()