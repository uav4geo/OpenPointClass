#include "scale.hpp"

Scale::Scale(size_t id, PointSet *pSet, double resolution, int kNeighbors, double radius) :
    id(id), pSet(pSet), resolution(resolution), kNeighbors(kNeighbors), radius(radius), scaledSet(new PointSet()){
    
}

void Scale::init(){
    #pragma omp critical
    {
        std::cout << "Init scale " << id << " at " << resolution << " ..." << std::endl;
    }
    if (id == 0){
        pSet->pointMap.resize(pSet->count());
    }else if (id > 0){
        eigenValues.resize(pSet->count());
        eigenVectors.resize(pSet->count());
        orderAxis.resize(pSet->count());
        heightMin.resize(pSet->count());
        heightMax.resize(pSet->count());

        if (id == 1){
            avgHsv.resize(pSet->count());
        }
    }

    computeScaledSet();
}

void Scale::build(){
    #pragma omp critical
    {
    std::cout << "Building scale " << id << " (" << scaledSet->count() << " points) ..." << std::endl;
    }

    #pragma omp parallel
    {
    KdTree *index = scaledSet->getIndex<KdTree>();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    std::vector<size_t> neighborIds(kNeighbors);
    std::vector<float> sqrDists(kNeighbors);

    #pragma omp for
    for (size_t idx = 0; idx < pSet->count(); idx++){
        index->knnSearch(&pSet->points[idx][0], kNeighbors, &neighborIds[0], &sqrDists[0]);
        Eigen::Vector3f medoid = computeMedoid(neighborIds);
        Eigen::Matrix3d covariance = computeCovariance(neighborIds, medoid);
        solver.computeDirect(covariance);
        Eigen::Vector3d ev = solver.eigenvalues();
        for (size_t i = 0; i < 3; i++) ev[i] = std::max(ev[i], 0.0);

        double sum = ev[0] + ev[1] + ev[2];
        eigenValues[idx] = (ev / sum).cast<float>(); // sum-normalized
        eigenVectors[idx] = solver.eigenvectors().cast<float>();

        // std::cout <<  "==Covariance==" << std::endl << 
        //     covariance << std::endl;
        // std::cout  << "==Medoid==" << std::endl << 
        //     medoid << std::endl;
        // std::cout <<  "==Eigenvalues==" << std::endl << 
        //     eigenValues[idx] << std::endl;
        // std::cout <<  "==Eigenvectors==" << std::endl << 
        //     eigenVectors[idx] << std::endl;
        // exit(1);

        // lambda1 = eigenValues[idx][2]
        // lambda3 = eigenValues[idx][0]

        // e1 = eigenVectors[idx].col(2)
        // e3 = eigenVectors[idx].col(0)
        orderAxis[idx](0,0) = 0.f;
        orderAxis[idx](1,0) = 0.f;
        orderAxis[idx](0,1) = 0.f;
        orderAxis[idx](1,1) = 0.f;

        heightMin[idx] = std::numeric_limits<float>::max();
        heightMax[idx] = std::numeric_limits<float>::min();
        
        for (size_t const &i : neighborIds){
            Eigen::Vector3f p(scaledSet->points[i][0],
                              scaledSet->points[i][1],
                              scaledSet->points[i][2]);
            Eigen::Vector3f n = (p - medoid);
            double v00 = n.dot(eigenVectors[idx].col(2));
            double v01 = n.dot(eigenVectors[idx].col(1));
            orderAxis[idx](0,0) += v00;
            orderAxis[idx](0,1) += v01;
            orderAxis[idx](1,0) += v00 * v00;
            orderAxis[idx](1,1) += v01 * v01;

            if (p[2] > heightMax[idx]) heightMax[idx] = p[2];
            if (p[2] < heightMin[idx]) heightMin[idx] = p[2];
        }
    }

    if (id == 1){
        std::vector<nanoflann::ResultItem<size_t, float>> radiusMatches;

        #pragma omp for
        for (size_t idx = 0; idx < pSet->count(); idx++){
            size_t numMatches = index->radiusSearch(&pSet->points[idx][0], radius, radiusMatches);
            avgHsv[idx] = {0.f, 0.f, 0.f};

            for (size_t i = 0; i < numMatches; i++){
                size_t nIdx = radiusMatches[i].first;
                // TODO: precompute HSV values at read time
                auto hsv = rgb2hsv(scaledSet->colors[nIdx][0], 
                                scaledSet->colors[nIdx][1], 
                                scaledSet->colors[nIdx][2]);
                for (size_t j = 0; j < 3; j++)
                    avgHsv[idx][j] += hsv[j];
            }

            if (numMatches > 0){
                for (size_t j = 0; j < 3; j++)
                    avgHsv[idx][j] /= numMatches;
            }
        }
    }

    }
}

void Scale::computeScaledSet(){
    if (scaledSet->points.size() == 0){
        bool trackPoints = id == 0;

        // Voxel centroid nearest neighbor
        // Roughly from https://raw.githubusercontent.com/PDAL/PDAL/master/filters/VoxelCentroidNearestNeighborFilter.cpp
        double x0 = pSet->points[0][0];
        double y0 = pSet->points[0][1];
        double z0 = pSet->points[0][2];

        // Make an initial pass through the input to index indices by
        // row, column, and depth.
        std::map<std::tuple<size_t, size_t, size_t>, std::vector<size_t> > populated_voxel_ids;

        for (size_t id = 0; id < pSet->count(); id++){
            populated_voxel_ids[std::make_tuple(
                static_cast<size_t>((pSet->points[id][0] - y0) / resolution),  // r
                static_cast<size_t>((pSet->points[id][1] - x0) / resolution),  // c
                static_cast<size_t>((pSet->points[id][2] - z0) / resolution) // d
            )].push_back(id);
        }

        // Make a second pass through the populated voxels to compute the voxel
        // centroid and to find its nearest neighbor.
        scaledSet->points.clear();
        scaledSet->colors.clear();

        for (auto const& t : populated_voxel_ids){
            if (t.second.size() == 1){
                // If there is only one point in the voxel, simply append it.
                scaledSet->appendPoint(*pSet, t.second[0]);
                if (trackPoints) scaledSet->trackPoint(*pSet, t.second[0]);
            }else if (t.second.size() == 2){
                // Else if there are only two, they are equidistant to the
                // centroid, so append the one closest to voxel center.

                // Compute voxel center.
                double y_center = y0 + (std::get<0>(t.first) + 0.5) * resolution;
                double x_center = x0 + (std::get<1>(t.first) + 0.5) * resolution;
                double z_center = z0 + (std::get<2>(t.first) + 0.5) * resolution;

                // Compute distance from first point to voxel center.
                double x1 = pSet->points[t.second[0]][0];
                double y1 = pSet->points[t.second[0]][1];
                double z1 = pSet->points[t.second[0]][2];
                double d1 = pow(x_center - x1, 2) + pow(y_center - y1, 2) + pow(z_center - z1, 2);

                // Compute distance from second point to voxel center.
                double x2 = pSet->points[t.second[1]][0];
                double y2 = pSet->points[t.second[1]][1];
                double z2 = pSet->points[t.second[1]][2];
                double d2 = pow(x_center - x2, 2) + pow(y_center - y2, 2) + pow(z_center - z2, 2);

                // Append the closer of the two.
                if (d1 < d2) scaledSet->appendPoint(*pSet, t.second[0]);
                else scaledSet->appendPoint(*pSet, t.second[1]);

                if (trackPoints){
                    scaledSet->trackPoint(*pSet, t.second[0]);
                    scaledSet->trackPoint(*pSet, t.second[1]);
                }
            } else  {
                // Else there are more than two neighbors, so choose the one
                // closest to the centroid.

                // Compute the centroid.
                Eigen::Vector3f centroid = computeCentroid(t.second);

                // Compute distance from each point in the voxel to the centroid,
                // retaining only the closest.
                size_t pmin = 0;
                double dmin((std::numeric_limits<double>::max)());
                for (auto const& p : t.second){
                    double sqr_dist = pow(centroid[0] - pSet->points[p][0], 2) +
                                    pow(centroid[1] - pSet->points[p][1], 2) +
                                    pow(centroid[2] - pSet->points[p][2], 2);
                    if (sqr_dist < dmin){
                        dmin = sqr_dist;
                        pmin = p;
                    }
                }

                scaledSet->appendPoint(*pSet, pmin);

                if (trackPoints){
                    for (auto const &p : t.second){
                        scaledSet->trackPoint(*pSet, p);
                    }
                }
            }
        }
    }

    if (id > 0) scaledSet->buildIndex<KdTree>();
}

void Scale::save(const std::string &filename){
    savePointSet(*scaledSet, filename);
}

Eigen::Matrix3d Scale::computeCovariance(const std::vector<size_t> &neighborIds, const Eigen::Vector3f &medoid){
    Eigen::MatrixXd A(3, neighborIds.size());
    size_t k = 0;

    for (size_t const &i : neighborIds){
        A(0, k) = scaledSet->points[i][0] - medoid[0];
        A(1, k) = scaledSet->points[i][1] - medoid[1];
        A(2, k) = scaledSet->points[i][2] - medoid[2];
        k++;
    }

    return A * A.transpose() / (neighborIds.size() - 1);
}

Eigen::Vector3f Scale::computeMedoid(const std::vector<size_t> &neighborIds){
    double mx, my, mz;
    mx = my = mz = 0.0;
    double minDist = std::numeric_limits<double>::max();
    for (size_t const &i : neighborIds){
        double sum = 0.0;
        double xi = scaledSet->points[i][0];
        double yi = scaledSet->points[i][1];
        double zi = scaledSet->points[i][2];

        for (size_t const &j : neighborIds){
            sum += pow(xi - scaledSet->points[j][0], 2) +
                   pow(yi - scaledSet->points[j][1], 2) +
                   pow(zi - scaledSet->points[j][2], 2);
        }

        if (sum < minDist){
            mx = xi;
            my = yi;
            mz = zi;
            minDist = sum;
        }
    }

    Eigen::Vector3f medoid;
    medoid << mx, my, mz;

    return medoid;
}

Eigen::Vector3f Scale::computeCentroid(const std::vector<size_t> &pointIds){
    double mx, my, mz;
    mx = my = mz = 0.0;
    size_t n = 0;
    for (auto const& j : pointIds){
        auto update = [&n](double value, double average){
            double delta, delta_n;
            delta = value - average;
            delta_n = delta / n;
            return average + delta_n;
        };
        n++;
        mx = update(pSet->points[j][0], mx);
        my = update(pSet->points[j][1], my);
        mz = update(pSet->points[j][2], mz);
    }

    Eigen::Vector3f centroid;
    centroid << mx, my, mz;

    return centroid;
}

std::vector<Scale *> computeScales(size_t numScales, PointSet *pSet, double startResolution, double radius){
    std::vector<Scale *> scales(numScales, nullptr);

    Scale *base = new Scale(0, pSet, startResolution * std::pow(2.0, 0), radius);
    base->init();
    // base->save("base.ply");
    pSet->base = base->scaledSet;

    for (size_t i = 0; i < numScales; i++){
        scales[i] = new Scale(i + 1, base->scaledSet, startResolution * std::pow(2.0, i), radius);
    }

    // Save some time on the first scale
    scales[0]->scaledSet = base->scaledSet;

    #pragma omp parallel for
    for (size_t i = 0; i < numScales; i++){
        scales[i]->init();
    }

    for (size_t i = 0; i < numScales; i++){
        scales[i]->build();
        // scales[i]->save("scale_" + std::to_string(i + 1) + ".ply");
    }

    return scales;
}
