# For PDE solving
from .experiments import run_pde_experiment
from .experiments import run_function_fitting_experiment,plot_function_fitting_results

def run_exp():
    run_pde_experiment()
    results = run_function_fitting_experiment(
        function_type='special',
        function_name='jv'
    )
    plot_function_fitting_results(results, 'Bessel Function')

if __name__ == '__main__':
    run_exp()