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

PD_REGISTER_KERNEL(conv2d, IPU, ALL_LAYOUT, phi::Conv2dKernel, float, double) {}

PD_REGISTER_KERNEL(pool2d, IPU, ALL_LAYOUT, phi::Pool2dKernel, float, double) {}
