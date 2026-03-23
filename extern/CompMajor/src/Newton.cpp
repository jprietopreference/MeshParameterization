#include "Newton.h"

#include <chrono>
#include <igl/flip_avoiding_line_search.h>
#include <mkl_types.h>


Newton::Newton() {
	pardiso.set_type(2, true);
	last_step_timing.analyze_pattern_time = 0.0;  // Initialize to 0
}

int Newton::step()
{
	auto eval_start = std::chrono::high_resolution_clock::now();
	eval_fgh(m_x, f, g, h);
	auto eval_end = std::chrono::high_resolution_clock::now();
	
	// Get the split timings from energy
	const auto& eval_timing = energy->get_last_eval_timing();
	last_step_timing.eval_value_time = eval_timing.value_time;
	last_step_timing.eval_gradient_time = eval_timing.gradient_time;
	last_step_timing.eval_hessian_time = eval_timing.hessian_time;

	SSd = energy->symDirichlet.SS;

	// auto prep_start = std::chrono::high_resolution_clock::now();
	SS.clear();
	SS.insert(SS.end(), SSd.begin(), SSd.end());
	auto prep_start = std::chrono::high_resolution_clock::now();
	pardiso.update_a(SS);
	auto prep_end = std::chrono::high_resolution_clock::now();
	last_step_timing.matrix_prep_time = std::chrono::duration<double>(prep_end - prep_start).count();

	auto factor_start = std::chrono::high_resolution_clock::now();
	// Try factorization, with regularization fallback on failure
	bool factored = false;
	for (double reg = 0; reg <= 1e2; reg = (reg == 0) ? 1e-6 : reg * 10) {
		if (reg > 0) {
			// Add regularization: H += reg * I
			for (size_t k = 0; k < SS.size(); k++) {
				if (II[k] == JJ[k]) SS[k] = SSd[k] + reg;
			}
			pardiso.update_a(SS);
		}
		try {
			pardiso.factorize();
			factored = true;
			break;
		} catch (std::exception &err) {
			if (reg >= 1e2) {
				cerr << "Factorization failed (reg=" << reg << "): " << err.what() << endl;
				return -1;
			}
		}
	}
	auto factor_end = std::chrono::high_resolution_clock::now();
	last_step_timing.factorization_time = std::chrono::duration<double>(factor_end - factor_start).count();

	std::cout << "Factorization succeeded" << std::endl;

	auto solve_start = std::chrono::high_resolution_clock::now();
	Vec rhs = -g;
	pardiso.solve(rhs, p);
	std::cout << "Solved Newton system; got p with " << p.rows() << " rows" << std::endl;
	auto solve_end = std::chrono::high_resolution_clock::now();
	last_step_timing.solve_time = std::chrono::duration<double>(solve_end - solve_start).count();

	return 0;
}


void Newton::initialize()
{
	eval_fgh(m_x, f, g, h);
	
	IId = energy->symDirichlet.II;
	JJd = energy->symDirichlet.JJ;
	SSd = energy->symDirichlet.SS;
    
    // Create sparse matrix (assuming you want a square matrix)
    MKL_INT n = *std::max_element(IId.begin(), IId.end()) + 1;

	// find pattern for initialization
	II.insert(II.end(), IId.begin(), IId.end());
	JJ.insert(JJ.end(), JJd.begin(), JJd.end());
	SS.insert(SS.end(), SSd.begin(), SSd.end());

	pardiso.set_pattern(II, JJ, SS);
	
	// Time the analyze_pattern call
	auto analyze_start = std::chrono::high_resolution_clock::now();
	pardiso.analyze_pattern();
	auto analyze_end = std::chrono::high_resolution_clock::now();
	last_step_timing.analyze_pattern_time = std::chrono::duration<double>(analyze_end - analyze_start).count();
}

void Newton::linesearch()
{
	Mat m_x2 = Eigen::Map<MatX2>(m_x.data(), m_x.rows() / 2, 2);
	Mat p2 = Eigen::Map<const MatX2>(p.data(), p.rows() / 2, 2);
	Mat m_plus_p = m_x2 + p2;
	std::function<double(Eigen::MatrixXd&)> ls_func = bind(&Newton::eval_ls, this, placeholders::_1);
	double alpha = igl::flip_avoiding_line_search(F, m_x2, m_plus_p, ls_func);
	m_x = Eigen::Map<Vec>(m_x2.data(), m_x2.rows() * m_x2.cols());
}

double Newton::eval_ls(Mat& x)
{
	double f;
	Vec g;
	Vec vec_x = Eigen::Map<Vec>(x.data(), x.rows()  * x.cols(), 1);
	eval_f(vec_x, f);
	return f;
}
