use four_bar::{efd::Curve as _, plot2d::IntoDrawingArea, *};

fn fft_recon(path: &[[f64; 2]], harmonic: usize) -> Vec<[f64; 2]> {
    use rustfft::{num_complex::Complex, num_traits::Zero as _};

    let mut data = path
        .iter()
        .map(|&[re, im]| Complex { re, im })
        .collect::<Vec<_>>();
    let len = data.len();
    let mut plan = rustfft::FftPlanner::new();
    plan.plan_fft_forward(len).process(&mut data);
    let n1 = harmonic / 2;
    let n2 = n1 + harmonic % 2;
    data.iter_mut()
        .take(len - n1)
        .skip(n2)
        .for_each(|c| c.set_zero());
    plan.plan_fft_inverse(len).process(&mut data);
    data.into_iter()
        .map(|c| c / len as f64)
        .map(|Complex { re, im }| [re, im])
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const PATH: &[&str] = &[
        "../four-bar-rs/syn-examples/crunode.closed.ron",
        "../four-bar-rs/syn-examples/cusp.closed.ron",
        "../four-bar-rs/syn-examples/c-shape.open.ron",
        "../four-bar-rs/syn-examples/sharp.open.ron",
        "../four-bar-rs/syn-examples/heart.closed.ron",
    ];
    let fb = ron::from_str::<FourBar>(&std::fs::read_to_string(PATH[4])?)?;
    let path = fb.curve(360);
    let path_closed = path.clone().closed_lin();
    let efd = efd::Efd2::from_curve_harmonic(&path_closed, None).unwrap();
    let harmonic = efd.harmonic();
    let fft_recon = fft_recon(&path_closed[..path_closed.len() - 1], harmonic * 2);
    let path_recon = efd.generate(180);
    let b = plot2d::SVGBackend::new("test.svg", (800 * 3, 800));
    let mut roots = b.into_drawing_area().split_evenly((1, 3));
    let [root1, root2, root3] = [roots.remove(0), roots.remove(0), roots.remove(0)];
    plot2d::plot(
        root1,
        [("", path.as_slice())],
        plot2d::Opt::from(&fb)
            .grid(false)
            .axis(false)
            .scale_bar(true),
    )?;
    let opt = plot2d::Opt::new()
        .grid(false)
        .font(50.)
        .dot(true)
        .legend(plot2d::LegendPos::MM);
    plot2d::plot(
        root2,
        [
            ("Original", path.as_slice()),
            ("EFD Reconstructed", path_recon.as_slice()),
        ],
        opt.clone(),
    )?;
    plot2d::plot(
        root3,
        [
            ("Original", path.as_slice()),
            ("FD Reconstructed", fft_recon.as_slice()),
        ],
        opt,
    )?;
    Ok(())
}
