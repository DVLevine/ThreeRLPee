#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "three_lp_matrices.hpp"
#include "stride_map.hpp"
#include "three_lp_params.hpp"

namespace py = pybind11;
using linalg::Mat;
using linalg::Vec;

class ThreeLPSim {
public:
    ThreeLPSim(double dt = 0.02, double t_ds = 0.1, double t_ss = 0.3, double max_action = 50.0)
        : dt_(dt), t_ds_(t_ds), t_ss_(t_ss), max_action_(max_action) {
        params_ = ThreeLPParams::Adult();
        state_ = Vec::Zero(12);
        phase_ = "ds";
        phase_time_ = 0.0;
        support_sign_ = 1.0;

        auto D_ds = discretize_phase(make3LP_DS(params_, t_ds_), dt_);
        enforce_ds_constraints(D_ds);
        D_ds_ = D_ds;

        auto D_ss = discretize_phase(make3LP_SS(params_, t_ss_, 1.0), dt_);
        enforce_ss_constraints(D_ss);
        D_ss_ = D_ss;
    }

    py::array_t<double> reset(py::array_t<double> state0 = py::array_t<double>()) {
        state_ = Vec::Zero(12);
        if (state0 && state0.size() > 0) {
            auto buf = state0.unchecked<1>();
            if (buf.shape(0) != 12) {
                throw std::runtime_error("state0 must have length 12");
            }
            for (ssize_t i = 0; i < buf.shape(0); ++i) {
                state_(i) = buf(i);
            }
        }
        phase_ = "ds";
        phase_time_ = 0.0;
        support_sign_ = 1.0;
        return to_array(state_);
    }

    std::pair<py::array_t<double>, py::dict> step(py::array_t<double> action) {
        auto act_buf = action.unchecked<1>();
        if (act_buf.shape(0) < 4) {
            throw std::runtime_error("action must have at least 4 entries (U block)");
        }
        const int act_dim = static_cast<int>(act_buf.shape(0));
        Vec r = Vec::Zero(13);
        for (int i = 0; i < std::min(act_dim, 4); ++i) {
            r(i) = clip(act_buf(i), -max_action_, max_action_);
        }
        if (act_dim >= 8) {
            for (int i = 0; i < 4; ++i) {
                r(4 + i) = clip(act_buf(4 + i), -max_action_, max_action_);
            }
        }
        r(12) = support_sign_;

        const auto& D = (phase_ == "ds") ? D_ds_ : D_ss_;
        state_ = D.A * state_ + D.B * r;

        phase_time_ += dt_;
        double current_duration = (phase_ == "ds") ? t_ds_ : t_ss_;
        while (phase_time_ >= current_duration && current_duration > 0.0) {
            phase_time_ -= current_duration;
            if (phase_ == "ds") {
                phase_ = "ss";
                current_duration = t_ss_;
            } else {
                phase_ = "ds";
                support_sign_ *= -1.0;
                current_duration = t_ds_;
            }
        }

        py::dict info;
        info["phase"] = phase_;
        info["phase_time"] = phase_time_;
        info["phase_duration"] = current_duration;
        info["support_sign"] = support_sign_;
        info["fallen"] = false;
        info["dt"] = dt_;

        return {to_array(state_), info};
    }

    // Closed-form stride propagation over one DS->SS phase with stance swap+recenter.
    std::pair<py::array_t<double>, py::dict> step_closed_form(py::array_t<double> action) {
        auto act_buf = action.unchecked<1>();
        if (act_buf.shape(0) < 8) {
            throw std::runtime_error("action must have length >= 8 (U + V)");
        }
        Vec U = Vec::Zero(4);
        Vec V = Vec::Zero(4);
        for (int i = 0; i < 4; ++i) {
            U(i) = clip(act_buf(i), -max_action_, max_action_);
            V(i) = clip(act_buf(4 + i), -max_action_, max_action_);
        }

        // Build phase maps for current support sign.
        auto DS = discretize_phase(make3LP_DS(params_, t_ds_), t_ds_);
        enforce_ds_constraints(DS);
        auto SS = discretize_phase(make3LP_SS(params_, t_ss_, support_sign_), t_ss_);
        enforce_ss_constraints(SS);
        auto stride = StrideMap::build(DS, SS);

        Mat C = recenter_to_stance_matrix();
        Mat S = S_Q();

        Mat A_step = C * S * stride.A;
        Mat B_step = C * S * stride.B;

        Vec R = Vec::Zero(B_step.cols());
        for (int i = 0; i < 4; ++i) {
            R(i) = U(i);
            R(4 + i) = V(i);
        }
        if (R.size() > 12) {
            R(12) = support_sign_;
        }

        state_ = A_step * state_ + B_step * R;

        // Swap stance
        support_sign_ *= -1.0;

        py::dict info;
        info["support_sign"] = support_sign_;
        info["phase"] = "ss";
        info["t_ds"] = t_ds_;
        info["t_ss"] = t_ss_;
        info["dt"] = dt_;
        info["fallen"] = false;
        return {to_array(state_), info};
    }

    py::array_t<double> get_state() const {
        return to_array(state_);
    }

    int state_dim() const { return 12; }
    int action_dim() const { return 8; }
    double dt() const { return dt_; }
    double t_ds() const { return t_ds_; }
    double t_ss() const { return t_ss_; }
    double support_sign() const { return support_sign_; }

private:
    static double clip(double v, double lo, double hi) {
        return std::max(lo, std::min(hi, v));
    }

    static py::array_t<double> to_array(const Vec& v) {
        return py::array_t<double>(v.size(), v.data());
    }

    ThreeLPParams params_;
    double dt_{};
    double t_ds_{};
    double t_ss_{};
    double max_action_{};

    Vec state_;
    std::string phase_;
    double phase_time_{};
    double support_sign_{};

    PhaseDiscrete D_ds_;
    PhaseDiscrete D_ss_;
};

PYBIND11_MODULE(three_lp_cpp, m) {
    py::class_<ThreeLPSim>(m, "ThreeLPSim")
        .def(py::init<double, double, double, double>(),
             py::arg("dt") = 0.02,
             py::arg("t_ds") = 0.1,
             py::arg("t_ss") = 0.3,
             py::arg("max_action") = 50.0)
        .def("reset", &ThreeLPSim::reset, py::arg("state0") = py::array_t<double>())
        .def("step", &ThreeLPSim::step)
        .def("step_closed_form", &ThreeLPSim::step_closed_form)
        .def("get_state", &ThreeLPSim::get_state)
        .def_property_readonly("state_dim", &ThreeLPSim::state_dim)
        .def_property_readonly("action_dim", &ThreeLPSim::action_dim)
        .def_property_readonly("dt", &ThreeLPSim::dt)
        .def_property_readonly("t_ds", &ThreeLPSim::t_ds)
        .def_property_readonly("t_ss", &ThreeLPSim::t_ss)
        .def_property_readonly("support_sign", &ThreeLPSim::support_sign);
}
