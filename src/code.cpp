#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include "emd.hpp"


//'@export
// [[Rcpp::export]]
void hello_world() {
  Rcpp::Rcout << "Hello" << std::endl;
  return;
}


arma::mat pairwise_dist(const arma::mat &x, const arma::mat &y, double p=2.0) {
  arma::mat out(x.n_rows, x.n_rows);

  #pragma omp parallel for
  for (int i = 0; i < x.n_rows; ++i) {
    arma::rowvec curr = x.row(i);
    out.row(i) = arma::sum(
      arma::pow(arma::abs(y.each_row() - curr), p), 1).t();
  }
  return arma::pow(out, 1.0/p);
}

//'@export
// [[Rcpp::export]]
arma::mat tranport_plan(const arma::mat &x_supp, arma::vec x_weight,
                        const arma::mat &y_supp, arma::vec y_weight, double p=2.0,
                        int max_iter=1000) {
  int n1 = x_weight.n_elem;
  int n2 = y_weight.n_elem;
  arma::vec alpha = arma::zeros<arma::vec>(n1);
  arma::vec beta = arma::zeros<arma::vec>(n2);
  arma::mat t_mat(n1, n2);
  int status = 0;

  double cost;

  // Compute pairwise distance (cost) matrix
  arma::mat cost_mat = pairwise_dist(x_supp, y_supp, p);

  status = EMD_wrap(
    n1, n1, x_weight.memptr(), y_weight.memptr(), cost_mat.memptr(),
    t_mat.memptr(), alpha.memptr(), beta.memptr(), &cost, max_iter);

  return t_mat;
}

//'@export
// [[Rcpp::export]]
arma::uvec match(const arma::mat &x, const arma::mat &y, double p=2.0, int max_iter=1000) {
  int n1 = x.n_rows;
  int n2 = y.n_rows;
  assert(n1 == n2);

  arma::vec w = arma::ones(n1) / n1;

  arma::mat plan = tranport_plan(x, w, y, w, p, max_iter);
  arma::umat perm_mat = arma::conv_to<arma::umat>::from(plan * n1);

  arma::uvec perm =  perm_mat * arma::regspace<arma::uvec>(
    0, n1 - 1);
  return perm + 1;
}

