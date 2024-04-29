#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <pybind11/stl.h>
#include <vector>
#include <numeric>
#include <algorithm>


/*
* reshape output to matrix *no need*
* try to not use memcpy for input *no need*
* test if using np.contiguous imrpoves performance *no need*
* eigen has a sum function which can be applied directly to numpy array input
* parallelize C++ Code
*/
namespace py = pybind11;

//py::array_t<double>
py::array_t<double> isocdf_seq(py::array_t<double>& w, py::array_t<double>& W, py::array_t<double>& Y, py::array_t<int>& posY, py::array_t<double>& y) {
    // Input
    std::vector<double> array_y(y.size());
    std::vector<double> array_w(w.size());

    // copy py::array -> std::vector
    std::memcpy(array_y.data(), y.data(), y.size() * sizeof(double));
    std::memcpy(array_w.data(), w.data(), w.size() * sizeof(double));

    double inf = std::numeric_limits<double>::infinity();

    //read array buffer input
    //py::buffer_info bufw = w.request(), bufW = W.request();
    py::buffer_info bufY = Y.request(), bufposY = posY.request(), bufW = W.request();

    //double* ptrw = (double*)bufw.ptr;
    double* ptrY = (double*)bufY.ptr;
    double* ptrW = (double*)bufW.ptr;
    int* ptrposY = (int*)bufposY.ptr;
    //double* ptry = (double*)bufy.ptr;



    py::ssize_t m = array_w.size();
    py::ssize_t mY = array_y.size();

    //int indxyMax = static_cast<int>(mY - 1);
    double yMax = array_y[mY-1];
   
    //double yMax = 1;
    // K is the maximal index such that Y[K] < Y[K + 1]
    py::ssize_t K = bufY.shape[0] - 1;
    while (ptrY[K] == ptrY[K - 1]) K--;
    K--;
    if (ptrY[K] < yMax) yMax = ptrY[K];

    // yInd is used to check when to store the CDF at y[yInd]
    int yInd = 0;
    array_y.push_back(inf);
    /*ptry.push_back(inf);*/
    while (array_y[yInd] < ptrY[0]) yInd++;

    // Prepare object containers
    std::vector<double> z0(m);
    std::vector<int> PP(m + 1);
    std::vector<double> WW(m + 1);
    std::vector<double> MM(m + 1);
    
    // Prepare output
    double *CDF = new double [m*mY];
    for (int i = 0; i < m * mY; i++) CDF[i] = 1;
    //std::vector<double>* CDF = new std::vector<double>[m*mY];

    //CDF.reserve(m * mY);
    /*
    py::array_t<double> cdf = py::array_t<double>(m*mY);
    py::buffer_info bufcdf = cdf.request();
    double* ptrcdf = (double*)bufcdf.ptr;*/
    PP[0] = -1;
    for (int i = 0; i < m * yInd; i++) CDF[i] = 0;
    int indxcdf = m * yInd;
    // First iteration
    int d = 1;
    int j0 = ptrposY[0]; // -1 FOR R INDEXING
    z0[j0] = ptrW[0] / array_w[j0];
    PP[1] = j0;
    //WW[1] = sum(ptrw, 0, j0 + 1);
    WW[1] = accumulate(array_w.begin(), array_w.begin() + j0 + 1, 0);
    MM[0] = inf;
    MM[1] = ptrW[0] / WW[1];
    if (j0 < m - 1) {
        d++;
        PP[2] = m - 1;
        //WW[2] = sum(ptrw, j0, m);
        WW[2] = accumulate(array_w.begin() + j0, array_w.begin() + m, 0);
    }
    int upper;
    int countloop;
    // Store results if needed; keep track of how many results have been stored
    if ((ptrY[0] < ptrY[1]) && (array_y[yInd] < ptrY[1])) {
        for (int l = 1; l <= d; l++) {
            int len = PP[l] - PP[l - 1];
            upper = indxcdf + len;
            double val = MM[l];
            for (int i = indxcdf; i < upper; i++) CDF[i] = val;
            indxcdf = upper;
        }
        yInd++;
        countloop = 1;
        while (array_y[yInd] < ptrY[1]) {
            upper = indxcdf + m;
            //startindx = len + indxcdf + countindx;
            for (int j = indxcdf; j < upper; j++) CDF[j] = CDF[j - m*countloop];
            //CDF.insert(CDF.end(), CDF.end() - m, CDF.end());
            indxcdf = upper;
            countloop++;
            yInd++;
        }
    }

    // Prepare objects for loop
    int k = 1;
    int b0;
    int a0;
    int s0;
    int dz;
    std::vector<int> remPP;
    std::vector<double> remWW;
    std::vector<double> remMM;
    
  
    while (ptrY[k] <= yMax) {
        // Update z0
        j0 = ptrposY[k]; // -1 FOR R INDEXING
        z0[j0] = z0[j0] + ptrW[k] / array_w[j0];

        // Find partition to which j0 belongs
        s0 = 0;
        while (j0 > PP[s0]) s0++;
        a0 = PP[s0 - 1] + 1;
        b0 = PP[s0];
        dz = d;

        // Copy tail of vector
        if (b0 < m - 1) {
            //remPP = PP[slice(s0 + 1, dz - s0 + 1, 1)];
            //remMM = MM[slice(s0 + 1, dz - s0 + 1, 1)];
            //remWW = WW[slice(s0 + 1, dz - s0 + 1, 1)];
            remPP.assign(PP.begin() + s0 + 1, PP.begin() + dz + 1);
            remMM.assign(MM.begin() + s0 + 1, MM.begin() + dz + 1);
            remWW.assign(WW.begin() + s0 + 1, WW.begin() + dz + 1);
        }

        // Update value on new partition
        d = s0;
        PP[s0] = j0;
        //WW[s0] = sum(ptrw, a0, j0 + 1);
        WW[s0] = accumulate(array_w.begin() + a0, array_w.begin() + j0 + 1, 0);
        MM[s0] = inner_product(array_w.begin() + a0, array_w.begin() + j0 + 1, z0.begin() + a0, 0.0) / WW[s0];
        //MM[s0] = innerproduct(ptrw, z0, a0, j0 + 1) / WW[s0];

        // Pooling
        while (MM[d - 1] <= MM[d]) {
            d--;
            MM[d] = WW[d] * MM[d] + WW[d + 1] * MM[d + 1];
            WW[d] = WW[d] + WW[d + 1];
            MM[d] = MM[d] / WW[d];
            PP[d] = PP[d + 1];
        }

        // Add new partitions, pool
        if (j0 < b0) {
            for (int i = j0 + 1; i <= b0; i++) {
                d++;
                PP[d] = i;
                WW[d] = array_w[i];
                MM[d] = z0[i];
                while (MM[d - 1] <= MM[d]) {
                    d--;
                    MM[d] = WW[d] * MM[d] + WW[d + 1] * MM[d + 1];
                    WW[d] = WW[d] + WW[d + 1];
                    MM[d] = MM[d] / WW[d];
                    PP[d] = PP[d + 1];
                }
            }
        }

        // Copy (if necessary)
        if (b0 < m - 1) {
            int l0 = dz - s0;
            for (int i = 0; i < l0; i++) {
                int pos = i + d + 1;
                PP[pos] = remPP[i];
                MM[pos] = remMM[i];
                WW[pos] = remWW[i];
            }
            d = d + l0;
        }

        // increase k
        k++;

        // Store in matrix (if necessary)
        if (ptrY[k - 1] < ptrY[k]) {
            while (array_y[yInd] < ptrY[k - 1]) yInd++;
            if (array_y[yInd] < ptrY[k]) {
                for (int l = 1; l <= d; l++) {
                    int len = PP[l] - PP[l - 1];
                    double val = MM[l];
                    upper = indxcdf + len;
                    for (int i = indxcdf; i < upper; i++) CDF[i] = val;
                    indxcdf = upper;
                }
                yInd++;
                countloop = 1;
                while (array_y[yInd] < ptrY[k]) {
                    upper = indxcdf + m;
                    for (int j = indxcdf; j < upper; j++) CDF[j] = CDF[j - m*countloop];
                    //CDF.insert(CDF.end(), CDF.end() - m, CDF.end());
                    yInd++;
                    countloop++;
                    indxcdf = upper;
                }
            }
        }
    }
    
    // Transform vector 'CDF' to matrix
    //if (array_y[mY - 1] >= ptrY[bufY.shape[0] - 1]) {
        //int len = m * mY - CDF.size();
        //for (int i = 0; i < len; i++) CDF.push_back(1.0);
    //}

    // CDF should be converted to a matrix with length(unique(y)) (mY)
    // columns (here, it is a vector where the columns are stacked).
    //py::array cdfout = py::cast(CDF);
    py::capsule free_when_done(CDF, [](void* f) {
        //auto foo = reinterpret_cast<std::vector<double>*>(f);
        double * foo = reinterpret_cast<double *>(f);
        delete foo;
        });
    return py::array_t<double>({ m*mY }, // shape
        { sizeof(double) },      // stride
        //CDF->data(),   // data pointer
        CDF,
        free_when_done);
}


py::array_t<double> compOrd_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> X) {
    auto X_buf = X.request();
    double* X_ptr = (double*) X_buf.ptr;
    int m = X.shape(0);
    int d = X.shape(1);
    py::array_t<double> M = py::array_t<double>({m, m});
    auto M_buf = M.request();
    double* M_ptr = (double*) M_buf.ptr;
    std::fill_n(M_ptr, m*m, 0.0);

    M_ptr[(m-1)*m + (m-1)] = 1;
    for (int i = 0; i < (m-1); i++) {
        M_ptr[i*m + i] = 1;
        for (int j = (i+1); j < m; j++) {
            std::vector<double> vec(2*d);
            std::vector<double> x1(d);
            std::vector<double> x2(d);
            for (int k = 0; k < d; k++) {
                vec[k] = x1[k] = X_ptr[i*d + k];
                vec[d+ k] = x2[k] = X_ptr[j*d + k];
            }

            std::sort(vec.begin(), vec.end());
            auto last = std::unique(vec.begin(), vec.end());
            vec.erase(last, vec.end());

            int nobs = vec.size();
            std::vector<double> F1(nobs);
            std::vector<double> F2(nobs);
            std::sort(x1.begin(), x1.end());
            std::sort(x2.begin(), x2.end());
            for (int l = 0; l < nobs; ++l) {
                F1[l] = std::upper_bound(x1.begin(), x1.end(), vec[l]) - x1.begin();
                F2[l] = std::upper_bound(x2.begin(), x2.end(), vec[l]) - x2.begin();
            }

            for (int l = 0; l < nobs; ++l) {
                F1[l] /= d;
                F2[l] /= d;
            }

            

            int test = 1;
            for (int k = 0; k < (nobs); k++) {
                if (F2[k] > F1[k]) {
                    test = 0;
                    break;
                }
            }
            if (test == 1) {
                M_ptr[i*m + j] = 1;
            }
            if (M_ptr[i*m + j] == 0) {
                int test2 = 1;
                for (int k = 0; k < (nobs); k++) {
                    if (F1[k] > F2[k]) {
                        test2 = 0;
                        break;
                    }
                }
                if (test2 == 1) {
                    M_ptr[j*m + i] = 1;
                }
            }
        }
    }
    return M;
}

py::list ecdf_comp_class_sd(py::array_t<double, py::array::c_style | py::array::forcecast> X, py::array_t<double, py::array::c_style | py::array::forcecast> t) {
    auto X_buf = X.request();
    auto X_ptr = static_cast<double *>(X_buf.ptr);
    auto X_shape = X.shape();
    int m = X_shape[0];
    int d = X_shape[1];

    std::vector<int> M_shape = {m, m};
    py::array_t<double> M(M_shape);
    auto M_buf = M.request();
    auto M_ptr = static_cast<double *>(M_buf.ptr);

    std::vector<int> classY(m);
    int class_count = 1;

    M_ptr[m * (m - 1) + (m - 1)] = 1;

    for (int i = 0; i < (m - 1); i++) {
        M_ptr[i * m + i] = 1;

        bool class_check = false;

        if (classY[i] == 0) {
            classY[i] = class_count;
            class_count += 1;
            class_check = true;
        }

        for (int j = (i + 1); j < m; j++) {
            std::vector<double> F1(d);
            std::vector<double> F2(d);
            int d2 = 1;
            int d3 = 1;
            bool check_equal = true;

            if (class_check == true) {
                for (int k = 0; k < d; k++) {
                    F1[k] = X_ptr[i * d + k];
                    F2[k] = X_ptr[j * d + k];

                    if (F1[k] != F2[k]) {
                        check_equal = false;
                    }

                    if (F1[k] < F2[k]) {
                        d2 = 0;
                    }

                    if (F1[k] > F2[k]) {
                        d3 = 0;
                    }
                }

                if (check_equal == true) {
                    classY[j] = class_count - 1;
                }
            } else {
                for (int k = 0; k < d; k++) {
                    F1[k] = X_ptr[i * d + k];
                    F2[k] = X_ptr[j * d + k];

                    if (F1[k] < F2[k]) {
                        d2 = 0;
                    }

                    if (F1[k] > F2[k]) {
                        d3 = 0;
                    }
                }
            }

            M_ptr[i * m + j] = d2;
            M_ptr[j * m + i] = d3;
        }
    }

    if (classY[m - 1] == 0) {
        classY[m - 1] = class_count;
    }

    py::list ret;
    ret.append(M);
    ret.append(classY);

    return ret;
}


py::list ecdf_list_comp_class_sd(py::list X, py::list t) {
    int m = X.size();
    py::array_t<double> M({m, m});
    int class_count = 1;
    auto M_ptr = M.mutable_unchecked<2>();
    std::vector<double> classY(m, 0);


    M_ptr(m-1, m-1) = 1;

    for (int i = 0; i < (m-1); i++) {
        M_ptr(i, i) = 1;
        bool class_check = false;
        if (classY[i] == 0){
            classY[i] = class_count;
            class_count += 1;
            class_check = true;
        }

        auto t1 = py::cast<py::array_t<double>>(t[i]);
        auto x1 = py::cast<py::array_t<double>>(X[i]);
        auto t1_ptr = t1.mutable_data(0);
        auto x1_ptr = x1.mutable_data(0);
        int t_l1 = t1.size();

        for (int j = (i+1); j < m; j++) {
            auto t2 = py::cast<py::array_t<double>>(t[j]);
            auto x2 = py::cast<py::array_t<double>>(X[j]);
            auto t2_ptr = t2.mutable_data(0);
            auto x2_ptr = x2.mutable_data(0);
            int t_l2 = t2.size();
            int dim_thresh = t_l1 + t_l2;
            py::array_t<double> thresholds({dim_thresh});
            auto thresholds_ptr = thresholds.mutable_data(0);

            for (int l = 0; l < t_l1; l++){
                thresholds_ptr[l] = t1_ptr[l];
            }
            for (int l = t_l1; l < dim_thresh; l++){
                thresholds_ptr[l] = t2_ptr[l - t_l1];
            }

            std::sort(thresholds_ptr, thresholds_ptr + dim_thresh);
            auto last = std::unique(thresholds_ptr, thresholds_ptr + dim_thresh);
            thresholds.resize({last - thresholds_ptr});

            int d = thresholds.size();
            py::array_t<double> F1({d}), F2({d});
            auto F1_ptr = F1.mutable_data(0);
            auto F2_ptr = F2.mutable_data(0);
            int d2 = 1, d3 = 1;
            bool check_equal = true;

            for (int k = 0; k < d; k++) {
                int indx1 = 0, indx2 = 0;

                while (indx1 < t_l1 && thresholds_ptr[k] >= t1_ptr[indx1]) {
                    indx1 += 1;
                }
                if (indx1 == 0) {
                    F1_ptr[k] = 0;
                } else {
                    F1_ptr[k] = x1_ptr[indx1 - 1];
                }

                while (indx2 < t_l2 && thresholds_ptr[k] >= t2_ptr[indx2]) {
                    indx2 += 1;
                }
                if (indx2 == 0) {
                    F2_ptr[k] = 0;
                } else {
                    F2_ptr[k] = x2_ptr[indx2 - 1];
                }

                if (F1_ptr[k] != F2_ptr[k]) {
                    check_equal = false;
                }
                if (F1_ptr[k] < F2_ptr[k]) {
                    d2 = 0;
                }
                if (F1_ptr[k] > F2_ptr[k]) {
                    d3 = 0;
                }
            }


            if (class_check == true && check_equal == true) {
                classY[j] = class_count - 1;
            }
            
            M_ptr(i, j) = d2;
            M_ptr(j, i) = d3;
        }
    }
    if (classY[m-1] == 0){
        classY[m-1] = class_count;
    }

    py::list ret;
    ret.append(M);
    ret.append(classY);
    return ret;
}



py::array_t<double> normal_comp_sd(py::array_t<double> X) {
    // Get the number of rows and columns of X
    ssize_t m = X.shape(0);

    // Create a NumericMatrix M with dimensions m x m
    py::array_t<double> M = py::array_t<double>({m, m});

    // Get pointers to the raw data of X and M
    auto X_ptr = X.unchecked<2>();
    auto M_ptr = M.mutable_unchecked<2>();

    // Fill in the values of M based on the conditions specified
    for (ssize_t i = 0; i < (m - 1); ++i) {
        M_ptr(i, i) = 1.0;
        for (ssize_t j = (i + 1); j < m; ++j) {
            if (std::abs(X_ptr(i, 1) - X_ptr(j, 1)) < 1e-9) {
                if (X_ptr(i, 0) >= X_ptr(j, 0)) {
                    M_ptr(j, i) = 1.0;
                }
                if (X_ptr(j, 0) >= X_ptr(i, 0)) {
                    M_ptr(i, j) = 1.0;
                }
            }
        }
    }

    return M;
}

py::array_t<double> normal_comp_sd_ab(py::array_t<double> X, double a, double b) {
    // Get the number of rows of X
    ssize_t m = X.shape(0);

    // Create a NumericMatrix M with dimensions m x m
    py::array_t<double> M = py::array_t<double>({m, m});

    // Get pointers to the raw data of X and M
    auto X_ptr = X.unchecked<2>();
    auto M_ptr = M.mutable_unchecked<2>();

    // Fill in the values of M based on the conditions specified
    for (ssize_t i = 0; i < (m - 1); ++i) {
        M_ptr(i, i) = 1.0;
        for (ssize_t j = (i + 1); j < m; ++j) {
            double diff = X_ptr(j, 1) - X_ptr(i, 1);
            if (std::abs(diff) < 1e-9) {
                if (X_ptr(i, 0) >= X_ptr(j, 0)) {
                    M_ptr(j, i) = 1.0;
                }
                if (X_ptr(j, 0) >= X_ptr(i, 0)) {
                    M_ptr(i, j) = 1.0;
                }
            } else {
                double num = X_ptr(i, 0) * X_ptr(j, 1) - X_ptr(j, 0) * X_ptr(i, 1);
                double test = num / diff;
                if (test < a || test > b) {
                    if (X_ptr(i, 0) >= X_ptr(j, 0)) {
                        M_ptr(j, i) = 1.0;
                    }
                    if (X_ptr(j, 0) >= X_ptr(i, 0)) {
                        M_ptr(i, j) = 1.0;
                    }
                }
            }
        }
    }

    return M;
}





py::list new_func2_sd(py::array_t<double, py::array::c_style> X, py::array_t<double, py::array::c_style> x) {    
    auto X_buf = X.request();
    auto x_buf = x.request();
    double* X_ptr = (double*) X_buf.ptr;
    double* x_ptr = (double*) x_buf.ptr;
    int mX = X.shape(0);
    int d = X.shape(1);
    int mx = x.shape(0);


    py::array_t<int> smaller_indx = py::array_t<int>({mx, mX}, 0);
    auto smaller_indx_buf = smaller_indx.request();
    int* smaller_indx_ptr = (int*) smaller_indx_buf.ptr;
    

    py::array_t<int> greater_indx = py::array_t<int>({mx, mX}, 0);
    auto greater_indx_buf = greater_indx.request();
    int* greater_indx_ptr = (int*) greater_indx_buf.ptr;

    
    for (int i = 0; i < mx; i++) {
        for (int j = 0; j < mX; j++) {
            smaller_indx_ptr[i*mX + j] = 0;
            greater_indx_ptr[i*mX + j] = 0;
            std::vector<double> vec(2*d);
            std::vector<double> x1(d);
            std::vector<double> x2(d);
            for (int k = 0; k < d; k++) {
                vec[k] = x1[k] = x_ptr[i*d + k];
                vec[d+ k] = x2[k] = X_ptr[j*d + k];
            }

            std::sort(vec.begin(), vec.end());
            auto last = std::unique(vec.begin(), vec.end());
            vec.erase(last, vec.end());

            int nobs = vec.size();
            std::vector<double> F1(nobs);
            std::vector<double> F2(nobs);
            std::sort(x1.begin(), x1.end());
            std::sort(x2.begin(), x2.end());
            for (int l = 0; l < nobs; ++l) {
                F1[l] = std::upper_bound(x1.begin(), x1.end(), vec[l]) - x1.begin();
                F2[l] = std::upper_bound(x2.begin(), x2.end(), vec[l]) - x2.begin();
            }

            for (int l = 0; l < nobs; ++l) {
                F1[l] /= d;
                F2[l] /= d;
            }

            
            int test = 1;
            for (int k = 0; k < (nobs); k++) {
                if (F2[k] > F1[k]) {
                    test = 0;
                    break;
                }
            }
            if (test == 1) {
                greater_indx_ptr[i*mX + j] = 1;
            }
            int test2 = 1;
            for (int k = 0; k < (nobs); k++) {
                if (F1[k] > F2[k]) {
                    test2 = 0;
                    break;
                }
            }
            if (test2 == 1) {
                smaller_indx_ptr[i*mX + j] = 1;
            }
            
        }
    }
    
    py::list ret;
    ret.append(smaller_indx);
    ret.append(greater_indx);
    return ret;
}


py::list new_func_list_sd(py::list X, py::list x, py::list gridx, py::list gridX) {    
    int mX = X.size();
    int mx = x.size();

    py::array_t<int> smaller_indx = py::array_t<int>({mx, mX}, 0);
    auto smaller_indx_buf = smaller_indx.request();
    int* smaller_indx_ptr = (int*) smaller_indx_buf.ptr;
    

    py::array_t<int> greater_indx = py::array_t<int>({mx, mX}, 0);
    auto greater_indx_buf = greater_indx.request();
    int* greater_indx_ptr = (int*) greater_indx_buf.ptr;

    
    for (int i = 0; i < mx; i++) {
        std::vector<double> x1 = py::cast<std::vector<double>>(x[i]);
        std::vector<double> t1 = py::cast<std::vector<double>>(gridx[i]);
        int t_l1 = t1.size();
        for (int j = 0; j < mX; j++) {
            smaller_indx_ptr[i*mX + j] = 0;
            greater_indx_ptr[i*mX + j] = 0;
            std::vector<double> x2 = py::cast<std::vector<double>>(X[j]);
            std::vector<double> t2 = py::cast<std::vector<double>>(gridX[j]);
            int t_l2 = t2.size();
            int dim_thresh = t_l1 + t_l2; 
            std::vector<double> thresholds(dim_thresh);

            for (int l = 0; l < t_l1; l++) {
                thresholds[l] = t1[l];
            }
            for (int l = t_l1; l < dim_thresh; l++) {
                thresholds[l] = t2[l-t_l1];
            }


            std::sort(thresholds.begin(), thresholds.end());
            auto last = std::unique(thresholds.begin(), thresholds.end());
            thresholds.erase(last, thresholds.end());

            int d = thresholds.size();
            std::vector<double> F1(d);
            std::vector<double> F2(d);

            for (int k = 0; k < d; k++) {
                int indx1 = 0;
                int indx2 = 0;
                while(indx1 < t_l1 && thresholds[k] >= t1[indx1]){
                    indx1 += 1;
                }
                if (indx1 == 0){
                    F1[k] = 0;
                } else {
                    F1[k] = x1[indx1 - 1];
                }
                while(indx2 < t_l2 && thresholds[k] >= t2[indx2]){
                    indx2 += 1;
                }
                if (indx2 == 0){
                    F2[k] = 0;
                } else {
                    F2[k] = x2[indx2-1];
                }
            }
            
            int test = 1;
            for (int k = 0; k < (d); k++) {
                if (F2[k] > F1[k]) {
                    test = 0;
                    break;
                }
            }
            if (test == 1) {
                greater_indx_ptr[i*mX + j] = 1;
            }
            int test2 = 1;
            for (int k = 0; k < d; k++) {
                if (F1[k] > F2[k]) {
                    test2 = 0;
                    break;
                }
            }
            if (test2 == 1) {
                smaller_indx_ptr[i*mX + j] = 1;
            }
            
        }
    }
    
    py::list ret;
    ret.append(smaller_indx);
    ret.append(greater_indx);
    return ret;
}


py::list indx_norm_sd(py::array_t<double, py::array::c_style> X, py::array_t<double, py::array::c_style> x) {    
    auto X_buf = X.request();
    auto x_buf = x.request();
    double* X_ptr = (double*) X_buf.ptr;
    double* x_ptr = (double*) x_buf.ptr;
    int mX = X.shape(0);
    int mx = x.shape(0);

    
    py::array_t<int> smaller_indx = py::array_t<int>({mx, mX}, 0);
    auto smaller_indx_buf = smaller_indx.request();
    int* smaller_indx_ptr = (int*) smaller_indx_buf.ptr;
    

    py::array_t<int> greater_indx = py::array_t<int>({mx, mX}, 0);
    auto greater_indx_buf = greater_indx.request();
    int* greater_indx_ptr = (int*) greater_indx_buf.ptr;

    for (int i = 0; i < mx; i++) {
        double sigma1 = x_ptr[i*2 + 1];
        double mu1 = x_ptr[i*2];
        for (int j = 0; j < mX; j++) {
            double sigma2 = X_ptr[j*2 + 1];
            double mu2 = X_ptr[j*2];
            smaller_indx_ptr[i*mX + j] = 0;
            greater_indx_ptr[i*mX + j] = 0;


            if (std::abs(sigma2 - sigma1) < 1e-9) {
                if (mu1 >= mu2) {  
                    smaller_indx_ptr[i*mX + j] = 1;
                }
                if (mu2 >= mu1) {
                    greater_indx_ptr[i*mX + j] = 1;
                }
            }
        }
    }

    py::list ret;
    ret.append(smaller_indx);
    ret.append(greater_indx);
    return ret;
}






PYBIND11_MODULE(_isodisSD, m) {
    m.def("isocdf_seq", &isocdf_seq);
    m.def("compOrd_cpp", &compOrd_cpp);
    m.def("ecdf_comp_class_sd", &ecdf_comp_class_sd);
    m.def("ecdf_list_comp_class_sd", &ecdf_list_comp_class_sd);
    m.def("normal_comp_sd", &normal_comp_sd);
    m.def("normal_comp_sd_ab", &normal_comp_sd_ab);
    m.def("new_func2_sd", &new_func2_sd);
    m.def("new_func_list_sd", &new_func_list_sd);
    m.def("indx_norm_sd", &indx_norm_sd);
}
/*
<%
setup_pybind11(cfg)
%>
*/







