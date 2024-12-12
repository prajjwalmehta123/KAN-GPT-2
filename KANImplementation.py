from kan.experiments import run_pde_experiment
from kan.experiments import run_function_fitting_experiment, plot_function_fitting_results


def main():
    # Run PDE experiment
    run_pde_experiment()

    # Run function fitting experiment
    results = run_function_fitting_experiment(
        function_type='special',
        function_name='jv'
    )
    plot_function_fitting_results(results, 'Bessel Function')


if __name__ == '__main__':
    main()