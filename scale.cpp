#include "scale.hpp"
#include "point_io.hpp"
#include <pdal/filters/VoxelCentroidNearestNeighborFilter.hpp>
#include <pdal/private/MathUtils.hpp>

Scale::Scale(size_t id, pdal::PointViewPtr pView, double resolution, int kNeighbors) :
    id(id), pView(pView), resolution(resolution), kNeighbors(kNeighbors){
    
    pdal::BufferReader reader;
    reader.addView(pView);

    pdal::VoxelCentroidNearestNeighborFilter filter;
    pdal::Options opts;
    opts.add("cell", resolution);
    filter.setOptions(opts);
    filter.setInput(reader);

    pdal::PointTable *table = new pdal::PointTable();
    filter.prepare(*table);
    auto pvSet = filter.execute(*table);
    pScaledView = *pvSet.begin();

    // Build KDTree
    kdTree = new pdal::KD3Index(*pScaledView);
    kdTree->build();

    eigenValues.resize(pView->size());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;

    for (pdal::PointId idx = 0; idx < pView->size(); idx++){
        pdal::PointRef p = pView->point(idx);
        
        // Compute eigen analysis structure
        pdal::PointIdList ids = kdTree->neighbors(p, kNeighbors);
        Eigen::Matrix3d covariance = computeCovariance(p, pScaledView, ids);
        
        solver.computeDirect(covariance, Eigen::ComputeEigenvectors);
        Eigen::Vector3d ev = solver.eigenvalues();
        
        double sum = ev[0] + ev[1] + ev[2]; 
        eigenValues[idx] = ev / sum; // sum-normalized
    }
}

void Scale::save(const std::string &filename) const{
    savePointSet(pScaledView, filename);
}

Eigen::Matrix3d Scale::computeCovariance(const pdal::PointRef &p, pdal::PointViewPtr neighborView, const pdal::PointIdList &neighborIds){
    size_t n = neighborIds.size() + 1;

    Eigen::Vector3d medoid = computeMedoid(p, neighborView, neighborIds);
    Eigen::MatrixXd A(3, n);
    size_t k = 0;
    for (auto const& i : neighborIds){
        pdal::PointRef np = neighborView->point(i);
        A(0, k) = np.getFieldAs<double>(pdal::Dimension::Id::X) - medoid[0];
        A(1, k) = np.getFieldAs<double>(pdal::Dimension::Id::Y) - medoid[1];
        A(2, k) = np.getFieldAs<double>(pdal::Dimension::Id::Z) - medoid[2];
        k++;
    }

    return A * A.transpose() / (neighborIds.size());
}

Eigen::Vector3d Scale::computeMedoid(const pdal::PointRef &p, pdal::PointViewPtr neighborView, const pdal::PointIdList &neighborIds){
    double minDist = std::numeric_limits<double>::infinity();
    Eigen::Vector3d medoid;

    std::vector<pdal::PointRef> points(neighborIds.size() + 1);
    size_t j = 0;
    points[j++] = p;
    for (auto const &i : neighborIds){
        points[j++] = neighborView->point(i);
    }

    for (pdal::PointRef &pi : points){
        double sum = 0.0;
        double xi = p.getFieldAs<double>(pdal::Dimension::Id::X);
        double yi = p.getFieldAs<double>(pdal::Dimension::Id::Y);
        double zi = p.getFieldAs<double>(pdal::Dimension::Id::Z);

        for (pdal::PointRef &p : points){
            double x = xi - p.getFieldAs<double>(pdal::Dimension::Id::X);
            double y = yi - p.getFieldAs<double>(pdal::Dimension::Id::Y);
            double z = zi - p.getFieldAs<double>(pdal::Dimension::Id::Z);
            sum += sqrt(x*x + y*y + z*z);
        }

        if (sum < minDist){
            medoid[0] = xi;
            medoid[1] = yi;
            medoid[2] = zi;
            minDist = sum;
        } 
    }

    return medoid;
}