#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
//#include <deal.II/lac/affine_constraints.h> 9.0.1


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace compstrip
{
using namespace dealii;


Point<2> Rot(double alpha, Point<2> point, Point<2> origine) //kronecker delta
{
	return (Point<2> ( origine(0) + std::cos(alpha)*(point(0)-origine(0)) - std::sin(alpha)*(point(1)-origine(1))  , origine(1) + std::sin(alpha)*(point(0)-origine(0)) + std::cos(alpha)*(point(1)-origine(1)) ));
}



inline double kd(int i, int j) //kronecker delta
{
	return (i==j ? 1.0 : 0.0);
}

class Mu : public Function<2>
{
	public:
		Mu(double mu0_in, double kappa_in) : Function<2> (1)
		{mu0 = mu0_in; kappa = kappa_in;};

		virtual double value(const Point<2> &p,const unsigned int component = 0) const;

		virtual void value_list (const std::vector<Point<2> > &points, std::vector<double> &value_list,const unsigned int component = 0) const;

	private:
		double mu0;
		double kappa;
};

double Mu::value(const Point<2> &p,const unsigned int /*component*/) const
{
	return (mu0 * exp(kappa*p(1)));
}

void Mu::value_list (const std::vector<Point<2> > &points,std::vector<double> &value_list,const unsigned int /*component*/) const
{
  const unsigned int n_points = points.size();
  Assert (value_list.size() == n_points, ExcDimensionMismatch (value_list.size(), n_points));
  for (unsigned int p=0; p<n_points; ++p)
	value_list[p] = value(points[p]);
}


class compstripProblem
{
	public:
        compstripProblem(double l, double e,double R, double joint_lenght,double a, double Lenght, double nu, double mu0, double kappa, unsigned int n, unsigned int n2, unsigned int refine);
		~compstripProblem();

		void set_dofs(py::array_t<double> x);
		py::array_t<double> F_resid();
		py::array_t<double> J_times_u(py::array_t<double> u);
		py::array_t<double> postprocesssolution(py::array_t<double> u);

		py::array_t<double> dof_x_coords();
		py::array_t<double> dof_y_coords();
		py::array_t<double> dof_x_or_y();

		py::array_t<double> get_vertx() {return vertx;};
		py::array_t<double> get_verty() {return verty;};
		py::array_t<unsigned int> get_vertu() {return vertu;};
		py::array_t<unsigned int> get_vertv() {return vertv;};
		//py::array_t<unsigned int> get_constrained_dofs(){return constrained_dofs;};

		void set_step_size(double ss) {step_size = ss;}
		void set_last_solution(py::array_t<double> ci);

	private:
		void assemble_system();
		void get_vertex_info();

		Triangulation<2> triangulation;

		DoFHandler<2> dof_handler;
		FESystem<2> fe;

		ConstraintMatrix constraints;
		SparsityPattern sparsity_pattern;
		SparseMatrix<double> system_matrix;

		Vector<double> present_solution;
		Vector<double> mult_vector;
		Vector<double> system_rhs;

		double l;
		double e;
		double R;
		double joint_lenght;
		double a;
		double Lenght;
		double nu;
		double mu0;
		double kappa;
		unsigned int n;
		unsigned int n2;

		bool isInitialized = false;
		bool isDataStale = true;

		py::array_t<double> vertx;
		py::array_t<double> verty;
		py::array_t<unsigned int> vertu;
		py::array_t<unsigned int> vertv;
		//py::array_t<unsigned int> constrained_dofs;

		double step_size;

		Vector<double> last_solution;

		Vector<double> present_solution_disp;

};

compstripProblem::compstripProblem(double l_in, double e_in,double R_in, double joint_lenght_in,double a_in, double Lenght_in, double nu_in, double mu0_in, double kappa_in, unsigned int n_in, unsigned int n2_in, unsigned int refine):
	dof_handler (triangulation),
	fe (FE_Q<2>(1),2),
	l(l_in),
	e(e_in),
	a(a_in),
	Lenght(Lenght_in),
	nu(nu_in),
	mu0(mu0_in),
	kappa(kappa_in),
	n(n_in),
	n2(n2_in),
	joint_lenght(joint_lenght_in),
	R(R_in)
{

	//unsigned int n2 = 2;
	//double joint_lenght = 0.2*l;
	//double R = 4*e;

	Triangulation<2> triangulation_low_low_rignt;
	Triangulation<2> triangulation_low_rignt;
	Triangulation<2> triangulation_low_joint1;
	Triangulation<2> triangulation_low_low_left;
	Triangulation<2> triangulation_low_left;
	Triangulation<2> triangulation_low_joint2;

	Triangulation<2> merge;

	double delta = 0.5*e/(std::cos(std::atan(2*a/Lenght)));

	const std::vector< unsigned int > &  	subdivisions_one_n {1,n};
	const std::vector< unsigned int > &  	subdivisions_n_one {n,1};
	const std::vector< unsigned int > &  	subdivisions_two_n {2,n};
	const std::vector< unsigned int > &  	subdivisions_n_two {n,2};

	const double xn = 0;
	const double yn = delta+joint_lenght ;
	Tensor<1,2> edgeNlr1;
	Tensor<1,2> edgeNlr2;
	Tensor<1,2> edgeNll1;
	Tensor<1,2> edgeNll2;
	const Point<2> N(xn,yn);
	edgeNlr1[0] = e/2.;	
	edgeNlr1[1] = 0;	
	edgeNlr2[0] = 0;
	edgeNlr2[1] = l*0.5-yn;
	edgeNll1[0] = -1*edgeNlr1[0];	
	edgeNll1[1] = edgeNlr1[1];	
	edgeNll2[0] = -1*edgeNlr2[0];
	edgeNll2[1] = edgeNlr2[1];
	const std::array< Tensor<1,2>,2> & edgesNlr {edgeNlr1,edgeNlr2};
	const std::array< Tensor<1,2>,2> & edgesNll {edgeNll2,edgeNll1};

	GridGenerator::subdivided_parallelepiped<2,2> (triangulation_low_rignt,N,edgesNlr,subdivisions_one_n, false);
	GridGenerator::subdivided_parallelepiped<2,2> (triangulation_low_left,N,edgesNll,subdivisions_n_one, false);

	const double xa = (yn)*Lenght*0.5/(std::sqrt(0.25*Lenght*Lenght+a*a));
	const double ya = (yn)*a/(std::sqrt(0.25*Lenght*Lenght+a*a))-delta;
	const Point<2> A(xa,ya);
	const Point<2> Aleft(-xa,ya);
	Tensor<1,2> edgeAlr1;
	Tensor<1,2> edgeAlr2;
	edgeAlr1[0] = Lenght/2.0 - xa;	
	edgeAlr1[1] = a-ya-delta;	
	edgeAlr2[0] = 0.;	
	edgeAlr2[1] = 2*delta;
	Tensor<1,2> edgeAll1;
	Tensor<1,2> edgeAll2;
	edgeAll1[0] = -1*edgeAlr1[0];	
	edgeAll1[1] = edgeAlr1[1];	
	edgeAll2[0] = -1*edgeAlr2[0];	
	edgeAll2[1] = edgeAlr2[1];

	const std::array< Tensor<1,2>,2> & edgesAlr {edgeAlr1,edgeAlr2};
	const std::array< Tensor<1,2>,2> & edgesAll {edgeAll2,edgeAll1};
	GridGenerator::subdivided_parallelepiped<2,2> (triangulation_low_low_rignt,A,edgesAlr,subdivisions_n_two, false);
	GridGenerator::subdivided_parallelepiped<2,2> (triangulation_low_low_left,Aleft,edgesAll,subdivisions_two_n, false);
	
	double temp = (yn-ya-2*delta)/(xa-0.5*e);
	double xr=0.5*(xa+0.5*e) + R*temp/std::sqrt(1+temp*temp) ;
	double yr=0.5*(ya+2*delta+yn) + R/std::sqrt(1+temp*temp);
	const Point<2> Rot_center(xr,yr);
	const Point<2> P(0.5*e,yn);

	std::vector<Point<2> > vertices (4*n2+2);
	const double PI  =3.141592653589793238463;

	const double ux=0.5*e-xr;
	const double uy=yn-yr;
	const double vx=xa-xr;
	const double vy=ya+2*delta-yr;
	const double alpha = std::acos( (ux*vx+uy*vy) / (std::sqrt(ux*ux+uy*uy)*std::sqrt(vx*vx+vy*vy))	); 
	printf("alpha %f",alpha);

	for (unsigned int i =0; i< 2*n2+1; i++ ) {
		vertices[i]=(Rot(i*alpha/(2*n2),P,Rot_center));
		printf("vertice %i x %f y %f",i,vertices[i][0],vertices[i][1]);			
	}
	for (unsigned int i =0; i<n2+1; i++ ) {		
		vertices[2*n2+1+i]=( Point<2> (0,yn-((yn+delta)*i/n2)) );	
		printf("vertice %i x %f y %f",2*n2+1+i,vertices[2*n2+1+i][0],vertices[2*n2+1+i][1]);		
	}

	for (unsigned int i =1; i< n2+1; i++ ) {		
		vertices[3*n2+1+i]=( Point<2> (xa*i/n2,-delta+(ya+delta)*i/n2) );
		printf("vertice %i x %f y %f",3*n2+1+i,vertices[3*n2+1+i][0],vertices[3*n2+1+i][1]);		
	}


	std::vector<CellData<2> > cells (2*n2, CellData<2>());
	for (unsigned int i=0; i<2*n2; ++i)
	{
	      	cells[i].vertices[0] = 2*n2+2+i;
	      	cells[i].vertices[1] = i+1;
	      	cells[i].vertices[2] = 2*n2+i+1;
	      	cells[i].vertices[3] = i;
	    	cells[i].material_id = 0;
		printf(" \n cell id %i vertices id %i  %i  %i %i",i,i,i+1,2*n2+i+1,2*n2+i+2);
	}

	triangulation_low_joint1.create_triangulation (vertices,cells,SubCellData());

	const Point<2> Rot_center_left(-xr,yr);
	const Point<2> P_left(-0.5*e,yn);

	std::vector<Point<2> > vertices_left (4*n2+2); 

	const double alpha_left = -alpha; 
	printf("alpha %f",alpha);

	for (unsigned int i =0; i< 2*n2+1; i++ ) {
		vertices_left[i]=(Rot(i*alpha_left/(2*n2),P_left,Rot_center_left));
		printf("vertice %i x %f y %f",i,vertices_left[i][0],vertices_left[i][1]);			
	}
	for (unsigned int i =0; i<n2+1; i++ ) {		
		vertices_left[2*n2+1+i]=( Point<2> (0,yn-((yn+delta)*i/n2)) );	
		printf("vertice %i x %f y %f",2*n2+1+i,vertices_left[2*n2+1+i][0],vertices_left[2*n2+1+i][1]);		
	}

	for (unsigned int i =1; i< n2+1; i++ ) {		
		vertices_left[3*n2+1+i]=( Point<2> (-1*xa*i/n2,-delta+(ya+delta)*i/n2) );
		printf("vertice %i x %f y %f",3*n2+1+i,vertices_left[3*n2+1+i][0],vertices_left[3*n2+1+i][1]);		
	}


	std::vector<CellData<2> > cells_left (2*n2, CellData<2>());
	for (unsigned int i=0; i<2*n2; ++i)
	{
	      	cells_left[i].vertices[0] = i;
	      	cells_left[i].vertices[1] = i+1;
	      	cells_left[i].vertices[2] = 2*n2+i+1;
	      	cells_left[i].vertices[3] = 2*n2+2+i;
	    	cells_left[i].material_id = 0;
		printf(" \n cell id %i vertices id %i  %i  %i %i",i,i,i+1,2*n2+i+1,2*n2+i+2);
	}

	printf("test");
	triangulation_low_joint2.create_triangulation (vertices_left,cells_left,SubCellData());
	printf("test");
	GridGenerator::merge_triangulations (triangulation_low_joint1,triangulation_low_low_rignt,triangulation_low_low_rignt);
	GridGenerator::merge_triangulations (triangulation_low_rignt,triangulation_low_low_rignt,triangulation_low_rignt);

	GridGenerator::merge_triangulations (triangulation_low_joint2,triangulation_low_low_left,triangulation_low_low_left);
	GridGenerator::merge_triangulations (triangulation_low_left,triangulation_low_low_left,triangulation_low_left);
	printf("test");
	GridGenerator::merge_triangulations (triangulation_low_left,triangulation_low_rignt,merge);
	printf("test");
 	Tensor<1,2> low_to_high;
	low_to_high[0]=0;
	low_to_high[1]=l;
	printf("test");
	Triangulation<2> triangulation_high;
	triangulation_high.copy_triangulation(merge);
	GridTools::rotate (PI,triangulation_high) ;	
	GridTools::shift (low_to_high,triangulation_high);
	GridGenerator::merge_triangulations (merge,triangulation_high,triangulation);

	//triangulation.refine_global(refine);
	printf("test");

  	std::ofstream out ("grid-7.eps");
  	GridOut grid_out;
  	grid_out.write_eps (triangulation, out);
 	std::cout << "Grid written to grid-7.eps" << std::endl;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	printf("begining setup");

	dof_handler.distribute_dofs(fe);
	present_solution.reinit(dof_handler.n_dofs()+4);
	mult_vector.reinit(dof_handler.n_dofs()+4);
	last_solution.reinit(dof_handler.n_dofs()+4);
	present_solution_disp.reinit(dof_handler.n_dofs());

	system_rhs.reinit(dof_handler.n_dofs()+4);

	double nverts = triangulation.n_used_vertices();
	vertx = py::array_t<double>(nverts);
	verty = py::array_t<double>(nverts);
	vertu = py::array_t<unsigned int>(nverts);
	vertv = py::array_t<unsigned int>(nverts);

	get_vertex_info();



	constraints.clear();
	{
		//DoFTools::make_hanging_node_constraints(dof_handler, constraints);
		DoFTools::make_hanging_node_constraints (dof_handler,constraints);
		unsigned int Low_right_1_x=0;
		unsigned int Low_right_1_y=0;
		unsigned int High_left_1_x=0;
		unsigned int High_left_1_y=0;
		unsigned int Low_right_2_x=0;
		unsigned int Low_right_2_y=0;
		unsigned int High_left_2_x=0;
		unsigned int High_left_2_y=0;
		unsigned int Low_right_3_x=0;
		unsigned int Low_right_3_y=0;
		unsigned int High_left_3_x=0;
		unsigned int High_left_3_y=0;

		unsigned int Low_left_1_x=0;
		unsigned int Low_left_1_y=0;
		unsigned int High_right_1_x=0;
		unsigned int High_right_1_y=0;
		unsigned int Low_left_2_x=0;
		unsigned int Low_left_2_y=0;
		unsigned int High_right_2_x=0;
		unsigned int High_right_2_y=0;
		unsigned int Low_left_3_x=0;
		unsigned int Low_left_3_y=0;
		unsigned int High_right_3_x=0;
		unsigned int High_right_3_y=0;




		typename DoFHandler<2>::cell_iterator
		cell = dof_handler.begin(),
		endc = dof_handler.end();
		for (; cell!=endc; ++cell)
		{

			for(unsigned int i=0; i< 4 ;++i)
			{
				
				
				Point<2> & coords = cell->vertex(i);
				printf("\n point %f %f",coords(0),coords(1));
				if(std::abs(xa + edgeAlr1[0]-coords(0))<1e-4 && std::abs(ya + edgeAlr1[1]-coords(1))<1e-4)
				{
					Low_right_3_x=cell->vertex_dof_index(i,0);
					Low_right_3_y=cell->vertex_dof_index(i,1);	
					printf("Low_right_3 found");
				}
				if(std::abs(-1*xa + edgeAll1[0]+edgeAll2[0]-coords(0))<1e-4 && std::abs(l-ya - edgeAll1[1]-edgeAll2[1]-coords(1))<1e-4)
				{
					High_left_3_x=cell->vertex_dof_index(i,0);
					High_left_3_y=cell->vertex_dof_index(i,1);	
					printf("High_left_3 found");	
				}
				if(std::abs(xa + edgeAlr1[0] +  0.5*edgeAlr2[0]-coords(0))<1e-4 && std::abs(ya + edgeAlr1[1] +  0.5*edgeAlr2[1]-coords(1))<1e-4)
				{
					Low_right_2_x=cell->vertex_dof_index(i,0);
					Low_right_2_y=cell->vertex_dof_index(i,1);	
					printf("Low_right_2 found");	
				}
				if(std::abs(-1*xa + edgeAll1[0]+0.5*edgeAll2[0]-coords(0))<1e-4 && std::abs(l-ya - edgeAll1[1]-0.5*edgeAll2[1]-coords(1))<1e-4)
				{
					High_left_2_x=cell->vertex_dof_index(i,0);
					High_left_2_y=cell->vertex_dof_index(i,1);	
					printf("High_left_2 found");	
				}
				if(std::abs(xa + edgeAlr1[0] +  edgeAlr2[0]-coords(0))<1e-4 && std::abs(ya + edgeAlr1[1] +  edgeAlr2[1]-coords(1))<1e-4)
				{
					Low_right_1_x=cell->vertex_dof_index(i,0);
					Low_right_1_y=cell->vertex_dof_index(i,1);	
					printf("Low_right_1 found");	
				}
				if(std::abs(-1*xa + edgeAll1[0]-coords(0))<1e-4 && std::abs(l-ya - edgeAll1[1]-coords(1))<1e-4)
				{
					High_left_1_x=cell->vertex_dof_index(i,0);
					High_left_1_y=cell->vertex_dof_index(i,1);	
					printf("High_left_1 found");
				}


				if(std::abs(-1*xa + edgeAll1[0]-coords(0))<1e-4 && std::abs(ya + edgeAll1[1]-coords(1))<1e-4)
				{
					Low_left_1_x=cell->vertex_dof_index(i,0);
					Low_left_1_y=cell->vertex_dof_index(i,1);	
					printf("Low_left_1 found");	
				}
				if(std::abs(xa +edgeAlr1[0]-coords(0))<1e-4 && std::abs(l-ya -edgeAlr1[1]-coords(1))<1e-4)
				{
					High_right_3_x=cell->vertex_dof_index(i,0);
					High_right_3_y=cell->vertex_dof_index(i,1);	
					printf("High_right_3 found");	
				}
				if(std::abs(-1*xa + edgeAll1[0] + 0.5*edgeAll2[0]-coords(0))<1e-4 && std::abs(ya + edgeAll1[1] + 0.5*edgeAll2[1]-coords(1))<1e-4)
				{
					Low_left_2_x=cell->vertex_dof_index(i,0);
					Low_left_2_y=cell->vertex_dof_index(i,1);	
					printf("Low_left_2 found");	
				}
				if(std::abs(xa + edgeAlr1[0]+0.5*edgeAlr2[0]-coords(0))<1e-4 && std::abs(l-ya - edgeAlr1[1]-0.5*edgeAlr2[1]-coords(1))<1e-4)
				{
					High_right_2_x=cell->vertex_dof_index(i,0);
					High_right_2_y=cell->vertex_dof_index(i,1);	
					printf("High_right_2 found");	
				}
				if(std::abs(-1*xa + edgeAll1[0] + edgeAll2[0]-coords(0))<1e-4 && std::abs(ya + edgeAll1[1] + edgeAll2[1]-coords(1))<1e-4)
				{
					Low_left_3_x=cell->vertex_dof_index(i,0);
					Low_left_3_y=cell->vertex_dof_index(i,1);	
					printf("Low_left_3 found");	
				}
				if(std::abs(xa +edgeAlr2[0] + edgeAlr1[0]-coords(0))<1e-4 && std::abs(l-ya -edgeAlr2[1] - edgeAlr1[1]-coords(1))<1e-4)
				{
					High_right_1_x=cell->vertex_dof_index(i,0);
					High_right_1_y=cell->vertex_dof_index(i,1);	
					printf("High_right_1 found");
				}

			}
		}

		//constraints.add_line (dof_handler.n_dofs());
		//constraints.add_line (dof_handler.n_dofs()+1);
     
		constraints.add_line (Low_right_1_x);
		constraints.add_entry (Low_right_1_x,High_left_1_x, 1);
		constraints.add_line (Low_right_1_y);
		constraints.add_entry (Low_right_1_y,High_left_1_y, 1);
		constraints.add_line (Low_right_2_x);
		constraints.add_entry (Low_right_2_x,High_left_2_x, 1);
		constraints.add_line (Low_right_2_y);
		constraints.add_entry (Low_right_2_y,High_left_2_y, 1);	
		constraints.add_line (Low_right_3_x);
		constraints.add_entry (Low_right_3_x,High_left_3_x, 1);
		constraints.add_line (Low_right_3_y);
		constraints.add_entry (Low_right_3_y,High_left_3_y, 1);

		constraints.add_line (Low_left_1_x);
		constraints.add_entry (Low_left_1_x,High_right_1_x, 1);
		constraints.add_line (Low_left_1_y);
		constraints.add_entry (Low_left_1_y,High_right_1_y, 1);
		constraints.add_line (Low_left_2_x);
		constraints.add_entry (Low_left_2_x,High_right_2_x, 1);
		constraints.add_line (Low_left_2_y);
		constraints.add_entry (Low_left_2_y,High_right_2_y, 1);
		constraints.add_line (Low_left_3_x);
		constraints.add_entry (Low_left_3_x,High_right_3_x, 1);
		constraints.add_line (Low_left_3_y);
		constraints.add_entry (Low_left_3_y,High_right_3_y, 1);

		constraints.print(std::cout);

		std::vector<bool> boundary_dofs (dof_handler.n_dofs(), false);
		DoFTools::extract_boundary_dofs (dof_handler, ComponentMask(),  boundary_dofs);


		const unsigned int first_boundary_dof = std::distance (boundary_dofs.begin(),std::find (boundary_dofs.begin(),boundary_dofs.end(),true));
		if(constraints.is_constrained(first_boundary_dof))
			printf("is_already_constrained, NOT good");
		else
			printf("not already constrained, Good");

		if(constraints.is_constrained(first_boundary_dof+1))
			printf("is_already_constrained, NOT good !!");
		else
			printf("not already constrained, Good");


		constraints.add_line (first_boundary_dof);
		constraints.add_line (first_boundary_dof+1);

		for (unsigned int i=first_boundary_dof+1; i<dof_handler.n_dofs(); ++i)
			{
				if(i%2==first_boundary_dof%2 && not constraints.is_constrained(i))
	    				constraints.add_entry (first_boundary_dof,i, -1);
				else if(i%2==(first_boundary_dof+1)%2 && not constraints.is_constrained(i))
	    				constraints.add_entry (first_boundary_dof+1,i, -1);
			}

		constraints.print(std::cout);
		printf("test");


 
		}
	constraints.close();
   
	printf(" setup constraints closed");
	{
		DynamicSparsityPattern dsp(dof_handler.n_dofs()+4);
		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		typename DoFHandler<2>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
		for (; cell!=endc; ++cell)
		{
			cell->get_dof_indices(local_dof_indices);
			std::vector<types::global_dof_index> local_dof_indices_p1(local_dof_indices);
			local_dof_indices_p1.push_back(dof_handler.n_dofs());
			local_dof_indices_p1.push_back(dof_handler.n_dofs()+1);
			local_dof_indices_p1.push_back(dof_handler.n_dofs()+2);
			local_dof_indices_p1.push_back(dof_handler.n_dofs()+3);
			constraints.add_entries_local_to_global(local_dof_indices_p1,dsp,true);
		}
		sparsity_pattern.copy_from(dsp);
		system_matrix.reinit(sparsity_pattern);
	} 


	printf("setup done \n");
	isInitialized = true;


	printf("compstripProblem object initialized...\n");
}

compstripProblem::~compstripProblem()
{
	dof_handler.clear();
}

py::array_t<double> compstripProblem::dof_x_or_y()
{
	auto result = py::array_t<double>(dof_handler.n_dofs());
	typename DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; v++)
		{
			for(unsigned int i=0; i<2; i++)
			{
				const unsigned int dof_index = cell->vertex_dof_index (v,i);
				*result.mutable_data(dof_index) = (double)i;
			}
		}
	}
	return result;
}

py::array_t<double> compstripProblem::dof_x_coords()
{
	auto result = py::array_t<double>(dof_handler.n_dofs());
	typename DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; v++)
		{
			Point<2> & coords = cell->vertex(v); //http://www.dealii.org/archive/dealii/msg02540.html
							     //https://www.dealii.org/8.4.0/doxygen/deal.II/classTriaAccessor.html#aec7b140271415e1b0a1cb15d9c860397

			for(unsigned int i=0; i<2; i++)
			{
				const unsigned int dof_index = cell->vertex_dof_index (v,i);
				*result.mutable_data(dof_index) = coords(0);
			}
		}
	}
	return result;
}
py::array_t<double> compstripProblem::dof_y_coords()
{
	auto result = py::array_t<double>(dof_handler.n_dofs());
	typename DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; v++)
		{
			Point<2> & coords = cell->vertex(v); //http://www.dealii.org/archive/dealii/msg02540.html
							     //https://www.dealii.org/8.4.0/doxygen/deal.II/classTriaAccessor.html#aec7b140271415e1b0a1cb15d9c860397

			for(unsigned int i=0; i<2; i++)
			{
				const unsigned int dof_index = cell->vertex_dof_index (v,i);
				*result.mutable_data(dof_index) = coords(1);
			}
		}
	}
	return result;
}

void compstripProblem::get_vertex_info()
{
	//unsigned int nverts = triangulation.n_used_vertices();
	typename DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; v++)
		{
			Point<2> & coords = cell->vertex(v); //http://www.dealii.org/archive/dealii/msg02540.html
							     //https://www.dealii.org/8.4.0/doxygen/deal.II/classTriaAccessor.html#aec7b140271415e1b0a1cb15d9c860397
			unsigned int gidx = cell->vertex_index(v);

			*vertx.mutable_data(gidx) = coords(0);
			*verty.mutable_data(gidx) = coords(1);
			*vertu.mutable_data(gidx) = cell->vertex_dof_index(v,0);
			*vertv.mutable_data(gidx) = cell->vertex_dof_index(v,1);
		}
	}
}

inline void dealii2pybind(Vector<double> & vin, py::array_t<double> & vout)
{
  for(int i=0;i<vout.size();i++)
    {
      *vout.mutable_data(i) = vin(i);
    }
}

inline void pybind2dealii(py::array_t<double> & vin, Vector<double> & vout)
{
  for(int i=0;i<vin.size();i++)
    {
      vout(i) = *vin.data(i);
    }
}

void compstripProblem::set_dofs(py::array_t<double> x)
{
	if(not isInitialized)
	{
		throw std::runtime_error("Must initialize first");
	}

	pybind2dealii(x,present_solution);
	constraints.distribute(present_solution);
	for(unsigned int idx=0;idx<dof_handler.n_dofs();idx++)
	{
		present_solution_disp(idx) = present_solution(idx);
	}

	isDataStale = true;

	return;
}

void compstripProblem::set_last_solution(py::array_t<double> ci)
{
	if(not isInitialized)
	{
		throw std::runtime_error("Must initialize first");
	}

	pybind2dealii(ci,last_solution);
	constraints.distribute(last_solution);
	isDataStale = true;

	return;
}

py::array_t<double> compstripProblem::F_resid()
{
	if(not isInitialized)
		throw std::runtime_error("Must initialize first");
//printf("avant assemble");
	if(isDataStale)
	{
		assemble_system();
	}
//printf("après assemble");
	auto result = py::array_t<double>(system_rhs.size());

	dealii2pybind(system_rhs,result);

	return result;
}

py::array_t<double> compstripProblem::postprocesssolution(py::array_t<double> u)
{
	if(not isInitialized)
	{
		throw std::runtime_error("Must initialize first");
	}

	if(isDataStale)
	{
		assemble_system();
	}

	pybind2dealii(u,mult_vector);

	constraints.distribute(mult_vector);
	auto result = py::array_t<double>(u.size());

	dealii2pybind(mult_vector,result);

	return result;
}

void compstripProblem::assemble_system()
{
  	//printf("begin assemble");
	const FEValuesExtractors::Vector displacements(0);
	Mu mu_func(mu0, kappa);
	const QGauss<2> quadrature_formula(2);
	const unsigned int n_q_points = quadrature_formula.size();
	std::vector<Tensor<2, 2> > solution_grads_u_total (n_q_points);
	Tensor<2,2> lambda;
	lambda[0][0] = present_solution(dof_handler.n_dofs());
	lambda[1][0] = present_solution(dof_handler.n_dofs()+1);
	lambda[0][1] = present_solution(dof_handler.n_dofs()+1);
	lambda[1][1] = present_solution(dof_handler.n_dofs()+2);
	const double load = present_solution(dof_handler.n_dofs()+3);



	system_matrix = 0;
	system_rhs = 0;

	isDataStale = false;

	FEValues<2> fe_values (fe, quadrature_formula, update_gradients | update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	std::vector<types::global_dof_index> local_j_lambda_index(1);
	local_j_lambda_index[0] = dof_handler.n_dofs();

	typename DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	double Acell = 0;

	//printf("begining of matrix computation");

	for (; cell!=endc; ++cell)
	{
		std::vector<double> mu_values(n_q_points);
		cell->get_dof_indices(local_dof_indices);
		Acell = cell->measure();
		fe_values.reinit(cell);

		//See Step 15
		fe_values[displacements].get_function_gradients(present_solution_disp,solution_grads_u_total);


		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
		{
			mu_func.value_list(fe_values.get_quadrature_points(), mu_values);
			const Tensor<2, 2> F = Physics::Elasticity::Kinematics::F(lambda + solution_grads_u_total[q_point]);
			const SymmetricTensor<2,2> C = Physics::Elasticity::Kinematics::C(F);
			const double I2 = third_invariant(C);
			const Tensor<2, 2> F_inv = invert(F);
			Tensor<2,2> dW_dF = mu_values[q_point]*(F-transpose(F_inv)+
								2.0*nu/(1.0-nu)*(I2-sqrt(I2))*
								transpose(F_inv));
			Tensor<4,2> d2W_dFdF;
			Tensor<2,2> d2W_dFdF00;
			
			//build d2W_dFdF

			for(unsigned int idx_i=0;idx_i<2;idx_i++)
			{
				for(unsigned int idx_j=0;idx_j<2;idx_j++)
				{

					for(unsigned int idx_k=0;idx_k<2;idx_k++)
					{
						for(unsigned int idx_l=0;idx_l<2;idx_l++)
						{
							d2W_dFdF[idx_i][idx_j][idx_k][idx_l] =
							mu_values[q_point] * (
							kd(idx_i,idx_k)*kd(idx_j,idx_l) +
							F_inv[idx_j][idx_k]*F_inv[idx_l][idx_i] -
							2.0*nu/(1.0-nu)*(I2-sqrt(I2))*F_inv[idx_j][idx_k]*F_inv[idx_l][idx_i] +
							4.0*nu/(1.0-nu)*(I2-0.5*sqrt(I2))*F_inv[idx_l][idx_k]*F_inv[idx_j][idx_i]);
						}
					}
					d2W_dFdF00[idx_i][idx_j] = d2W_dFdF[idx_i][idx_j][0][0];
				}
			}

			for(unsigned int i=0;i<dofs_per_cell; ++i)
			{
				unsigned int I = local_dof_indices[i];
				const unsigned int component_i = fe.system_to_component_index(i).first;

				for(unsigned int j=0;j<dofs_per_cell; ++j)
				{
					unsigned int J = local_dof_indices[j];
					const unsigned int component_j = fe.system_to_component_index(j).first;
					//cell_matrix(i,j) += contract3(fe_values[displacements].gradient(i,q_point),d2W_dFdF,fe_values[displacements].gradient(j,q_point))*fe_values.JxW(q_point);
					for(unsigned int idx_p=0;idx_p<2;idx_p++)
					{
						for(unsigned int idx_l=0;idx_l<2;idx_l++)
						{
							//build Jij
							system_matrix.add(I,J,d2W_dFdF[component_i][idx_p][component_j][idx_l] *
																	fe_values.shape_grad(j,q_point)[idx_l] *
																	fe_values.shape_grad(i,q_point)[idx_p] *
																	fe_values.JxW(q_point));
						}
					}
				}

				for(unsigned int idx_p=0;idx_p<2;idx_p++)
				{
					system_rhs(I) += (dW_dF[component_i][idx_p] + (kd(component_i,1)*kd(idx_p,1)+kd(component_i,0)*kd(idx_p,0))*Acell*load)*
												fe_values.shape_grad(i,q_point)[idx_p] *
												fe_values.JxW(q_point);


					//cell_j_lambda
					system_matrix.add(I,dof_handler.n_dofs(),d2W_dFdF[0][0][component_i][idx_p] *
												fe_values.shape_grad(i,q_point)[idx_p] *
												fe_values.JxW(q_point));

					system_matrix.add(I,dof_handler.n_dofs()+1,(d2W_dFdF[0][1][component_i][idx_p] + d2W_dFdF[1][0][component_i][idx_p]) *
												fe_values.shape_grad(i,q_point)[idx_p] *
												fe_values.JxW(q_point));

					system_matrix.add(I,dof_handler.n_dofs()+2,d2W_dFdF[1][1][component_i][idx_p] *
												fe_values.shape_grad(i,q_point)[idx_p] *
												fe_values.JxW(q_point));

					system_matrix.add(I,dof_handler.n_dofs()+3,Acell*(kd(component_i,1)*kd(idx_p,1)+kd(component_i,0)*kd(idx_p,0)) * fe_values.shape_grad(i,q_point)[idx_p] *
												fe_values.JxW(q_point));

					system_matrix.add(dof_handler.n_dofs()+2,I,(d2W_dFdF[component_i][idx_p][1][1])*
												fe_values.shape_grad(i,q_point)[idx_p] *
												fe_values.JxW(q_point));

					system_matrix.add(dof_handler.n_dofs()+1,I,(d2W_dFdF[component_i][idx_p][0][1] + d2W_dFdF[component_i][idx_p][1][0]) *
												fe_values.shape_grad(i,q_point)[idx_p] *
												fe_values.JxW(q_point));

					system_matrix.add(dof_handler.n_dofs(),I,(d2W_dFdF[component_i][idx_p][0][0])*
												fe_values.shape_grad(i,q_point)[idx_p] *
												fe_values.JxW(q_point));


				}





			}

			system_matrix.add(dof_handler.n_dofs(),dof_handler.n_dofs(),d2W_dFdF[0][0][0][0] * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs(),dof_handler.n_dofs()+1,(d2W_dFdF[0][1][0][0]+d2W_dFdF[1][0][0][0]) * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs(),dof_handler.n_dofs()+2,d2W_dFdF[1][1][0][0] * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs(),dof_handler.n_dofs()+3, Acell * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs()+1,dof_handler.n_dofs(),(d2W_dFdF[0][0][0][1] + d2W_dFdF[0][0][1][0]) * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs()+1,dof_handler.n_dofs()+1,(d2W_dFdF[0][1][0][1] + d2W_dFdF[0][1][1][0] + d2W_dFdF[1][0][0][1] + d2W_dFdF[1][0][1][0]) *fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs()+1,dof_handler.n_dofs()+2,(d2W_dFdF[1][1][0][1] + d2W_dFdF[1][1][1][0]) * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs()+1,dof_handler.n_dofs()+3, 0*fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs()+2,dof_handler.n_dofs(),d2W_dFdF[0][0][1][1] * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs()+2,dof_handler.n_dofs()+1,(d2W_dFdF[0][1][1][1] + d2W_dFdF[1][0][1][1]) * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs()+2,dof_handler.n_dofs()+2,d2W_dFdF[1][1][1][1] * fe_values.JxW(q_point));
			system_matrix.add(dof_handler.n_dofs()+2,dof_handler.n_dofs()+3, Acell * fe_values.JxW(q_point));

			system_rhs(dof_handler.n_dofs()) += (dW_dF[0][0] +Acell*load)* fe_values.JxW(q_point);
			system_rhs(dof_handler.n_dofs()+1) += (dW_dF[1][0]+dW_dF[0][1])* fe_values.JxW(q_point);
			system_rhs(dof_handler.n_dofs()+2) += (dW_dF[1][1] +Acell*load)* fe_values.JxW(q_point);
		}
	}


	Vector<double> last_row = present_solution;// - last_solution;
	last_row -= last_solution;

	for(unsigned int i=0;i<dof_handler.n_dofs()+4;i++)
	{
		system_matrix.add(dof_handler.n_dofs()+3,i,last_row(i));
	}
	

	system_rhs(dof_handler.n_dofs()+3) = 0.5*(pow(last_row.l2_norm(),2.0)-pow(step_size,2.0));
	//system_rhs(dof_handler.n_dofs()) = 0; //U
	//system_rhs(dof_handler.n_dofs()+1) = 0; //U
	//system_rhs(dof_handler.n_dofs()+2) = 0; //U

	constraints.condense(system_matrix,system_rhs);
	//printf("\n fin assemble");
}

py::array_t<double> compstripProblem::J_times_u(py::array_t<double> u)
{
	//printf("begining J_times_u");
	if(not isInitialized)
		throw std::runtime_error("Must initialize first");

	if(isDataStale)
	{
		assemble_system();
	}

	pybind2dealii(u,mult_vector);

	Vector<double> tempsoln(u.size());
	system_matrix.vmult(tempsoln,mult_vector);

	auto result = py::array_t<double>(u.size());
	dealii2pybind(tempsoln,result);

	return result;
}
}


PYBIND11_MODULE(compstrip, m) {
	py::class_<compstrip::compstripProblem>(m, "compstripProblem",py::buffer_protocol())
	 .def(py::init<double, double,double, double,double, double, double, double, double, unsigned int,unsigned int, unsigned int>
	  (),py::arg("l_in"), py::arg("e_in"),py::arg("R_in"), py::arg("joint_lenght_in"),py::arg("a_in"), py::arg("Lenght_in"), py::arg("nu_in"), py::arg("mu0_in"),py::arg("kappa_in"), py::arg("n_in"), py::arg("n2_in"), py::arg("refine"))
		.def("set_dofs",&compstrip::compstripProblem::set_dofs)
		.def("F_resid",&compstrip::compstripProblem::F_resid)
		.def("postprocesssolution",&compstrip::compstripProblem::postprocesssolution)
		.def("dof_x_coords",&compstrip::compstripProblem::dof_x_coords)
		.def("dof_y_coords",&compstrip::compstripProblem::dof_y_coords)
		.def("dof_x_or_y",&compstrip::compstripProblem::dof_x_or_y)
		.def("get_vertx",&compstrip::compstripProblem::get_vertx)
		.def("get_verty",&compstrip::compstripProblem::get_verty)
		.def("get_vertu",&compstrip::compstripProblem::get_vertu)
		.def("get_vertv",&compstrip::compstripProblem::get_vertv)
		.def("set_step_size",&compstrip::compstripProblem::set_step_size)
		.def("set_last_solution",&compstrip::compstripProblem::set_last_solution)
		//.def("get_constrained_dofs",&compstrip::compstripProblem::get_constrained_dofs)
		.def("get_J_times_u",&compstrip::compstripProblem::J_times_u);
}

