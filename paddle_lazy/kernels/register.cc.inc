PD_REGISTER_KERNEL(abs,
                   IPU,
                   ALL_LAYOUT,
                   phi::AbsKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(conv2d, IPU, ALL_LAYOUT, phi::Conv2DKernel, float, double) {}

PD_REGISTER_KERNEL(pool2d, IPU, ALL_LAYOUT, phi::Pool2DKernel, float, double) {}

PD_REGISTER_KERNEL(sin, IPU, ALL_LAYOUT, phi::SinKernel, float, double) {}

PD_REGISTER_KERNEL(mean, IPU, ALL_LAYOUT, phi::MeanKernel, float, double) {}

PD_REGISTER_KERNEL(
    mean_grad, IPU, ALL_LAYOUT, phi::MeanGradKernel, float, double) {}

PD_REGISTER_KERNEL(
    conv2d_grad, IPU, ALL_LAYOUT, phi::Conv2DGradKernel, float, double) {}

PD_REGISTER_KERNEL(abs_grad,
                   IPU,
                   ALL_LAYOUT,
                   phi::AbsGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(
    sin_grad, IPU, ALL_LAYOUT, phi::SinGradKernel, float, double) {}
