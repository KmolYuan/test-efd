use four_bar::*;

fn fft_recon(path: &[[f64; 2]], harmonic: usize) -> Vec<[f64; 2]> {
    use ndarray::{arr1, s, Array};
    use ndrustfft::{Complex, Zero as _};

    let data = path
        .iter()
        .map(|&[re, im]| Complex { re, im })
        .collect::<Vec<_>>();
    let len = data.len();
    let mut plan = ndrustfft::FftHandler::new(len);
    let data = arr1(&data);
    let mut fd = Array::zeros(data.raw_dim());
    ndrustfft::ndfft_par(&data, &mut fd, &mut plan, 0);
    let n1 = harmonic / 2;
    let n2 = n1 + harmonic % 2;
    fd.slice_mut(s![n1..len - n2])
        .iter_mut()
        .for_each(|c| c.set_zero());
    let mut data = Array::zeros(fd.raw_dim());
    ndrustfft::ndifft_par(&fd, &mut data, &mut plan, 0);
    data.into_iter()
        .map(|Complex { re, im }| [re, im])
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fb = ron::from_str::<FourBar>(&std::fs::read_to_string(
        "../four-bar-rs/syn-examples/crunode.closed.ron",
    )?)?;
    let path = fb.curve(360);
    let efd = efd::Efd2::from_curve_harmonic(curve::closed_lin(&path), None).unwrap();
    let harmonic = efd.harmonic();
    let fft_recon = fft_recon(&path, harmonic);
    let path_recon = efd.generate(180);
    let b = plot2d::SVGBackend::new("test.svg", (800, 800));
    plot2d::plot(
        b,
        [
            ("Original", path.as_slice()),
            // (
            //     &format!("EFD Reconstructed ({harmonic} harmonics)"),
            //     path_recon.as_slice(),
            // ),
            (
                &format!("FD Reconstructed ({harmonic} harmonics)"),
                fft_recon.as_slice(),
            ),
        ],
        plot2d::Opt::from(None)
            .grid(false)
            // .axis(false)
            .dot(true)
            .stroke(4),
    )?;
    Ok(())
}
