import streamlit as st

#App title/ description
st.header('About DeepSolarEye')
st.header(':mostly_sunny: Solar Panel Power Loss Estimator :mostly_sunny:')

st.write('DeepSolarEye investigating the effects that soiling has on the power loss for solar panels.  \n\
    An RGB image of a solar panel combined with time and irradiance input predicts the power loss.  \n\
    The model has been trained on data collected by Cornell University and is based on the following research paper:  \n\
\n\
    \n\
    S. Mehta, A. P. Azad, S. A. Chemmengath, V. Raykar and S. Kalyanaraman, \n\
    DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization  \n\
    via Fully Convolutional Networks for Solar Panels," \n\
    2018 IEEE Winter Conference on Applications of Computer Vision (WACV),  \n\
    Lake Tahoe, NV, 2018, pp. 333-342.')
