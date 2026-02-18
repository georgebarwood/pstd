use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

criterion_group!(
    benches,
    bench_small_iter,
    bench_get,
    bench_sget,
    bench_clone,
    bench_ref_iter,
    bench_into_iter,
    bench_split_off,
    bench_insert
);
criterion_main!(benches);

fn bench_small_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("Small Iter");
    for n in [1, 10, 60, 1000].iter() {
        let mut exp_map = pstd::collections::BTreeMap::new();
        for i in 0..*n {
            exp_map.insert(i, i);
        }

        let mut std_map = std::collections::BTreeMap::new();
        for i in 0..*n {
            std_map.insert(i, i);
        }

        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| for _i in exp_map.iter() {})
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| for _i in std_map.iter() {})
        });
    }
    group.finish();
}

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("Clone");
    for n in [1000, 10000].iter() {
        let mut exp_map = pstd::collections::BTreeMap::new();
        for i in 0..*n {
            exp_map.insert(i, i);
        }

        let mut std_map = std::collections::BTreeMap::new();
        for i in 0..*n {
            std_map.insert(i, i);
        }

        group.bench_function(BenchmarkId::new("Exp", n), |b| b.iter(|| exp_map.clone()));
        group.bench_function(BenchmarkId::new("Std", n), |b| b.iter(|| std_map.clone()));
    }
    group.finish();
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("Insert");
    for n in [1000, 10000].iter() {
        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| {
                let mut m = pstd::collections::BTreeMap::new();
                for i in 0..*n {
                    m.insert(i, i);
                }
                assert!(m.len() == *n);
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| {
                let mut m = std::collections::BTreeMap::new();
                for i in 0..*n {
                    m.insert(i, i);
                }
                assert!(m.len() == *n);
            })
        });
    }
    group.finish();
}

fn bench_sget(c: &mut Criterion) {
    //use rand::Rng;
    //let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("Sget");
    for n in [10, 20, 50, 100, 200, 500, 1000].iter() {
        let n = *n;
        let mut s = Vec::new();
        for i in 0..n {
            s.push(i.to_string());
        }
        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            let mut map = pstd::collections::BTreeMap::new();
            for i in 0..n {
                map.insert(i.to_string(), i.to_string());
            }
            b.iter(|| {
                for i in 0..n {
                    /* let ri = rng.gen::<usize>() % n; */
                    assert!(map.get(&s[i]).unwrap() == &s[i]);
                }
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            let mut map = std::collections::BTreeMap::new();
            for i in 0..n {
                map.insert(i.to_string(), i.to_string());
            }
            b.iter(|| {
                for i in 0..n {
                    /* let ri = rng.gen::<usize>() % n; */
                    assert!(map.get(&s[i]).unwrap() == &s[i]);
                }
            })
        });
    }
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("Get");
    for n in [10, 20, 50, 100, 200, 500, 1000].iter() {
        let n = *n;
        let mut exp_map = pstd::collections::BTreeMap::new();
        for i in 0..n {
            exp_map.insert(i, i);
        }

        let mut std_map = std::collections::BTreeMap::new();
        for i in 0..n {
            std_map.insert(i, i);
        }

        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| {
                for i in 0..n {
                    assert!(exp_map.get(&i).unwrap() == &i);
                }
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| {
                for i in 0..n {
                    assert!(std_map.get(&i).unwrap() == &i);
                }
            })
        });
    }
    group.finish();
}

fn bench_ref_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("RefIter");
    for n in [50, 100, 1000, 10000, 100000].iter() {
        let mut exp_map = pstd::collections::BTreeMap::new();
        for i in 0..*n {
            exp_map.insert(i, i);
        }

        let mut std_map = std::collections::BTreeMap::new();
        for i in 0..*n {
            std_map.insert(i, i);
        }

        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| {
                for (k, v) in exp_map.iter() {
                    assert!(k == v);
                }
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| {
                for (k, v) in std_map.iter() {
                    assert!(k == v);
                }
            })
        });
    }
    group.finish();
}

fn exp_into_iter_test(n: usize) {
    let mut m = pstd::collections::BTreeMap::<usize, usize>::default();
    for i in 0..n {
        m.insert(i, i);
    }
    for (k, v) in m {
        assert!(k == v);
    }
}

fn std_into_iter_test(n: usize) {
    let mut m = std::collections::BTreeMap::<usize, usize>::default();
    for i in 0..n {
        m.insert(i, i);
    }
    for (k, v) in m {
        assert!(k == v);
    }
}

fn bench_into_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("IntoIter");
    for n in [100, 1000, 10000].iter() {
        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| {
                exp_into_iter_test(*n);
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| {
                std_into_iter_test(*n);
            })
        });
    }
    group.finish();
}

fn exp_split_off_test(n: usize) {
    let mut m = pstd::collections::BTreeMap::<usize, usize>::new();
    for i in 0..n {
        m.insert(i, i);
    }
    assert!(m.len() == n);
    let m2 = m.split_off(&(n / 2));
    assert!(m2.len() == n / 2);
    assert!(m.len() + m2.len() == n);
}
fn std_split_off_test(n: usize) {
    let mut m = std::collections::BTreeMap::<usize, usize>::new();
    for i in 0..n {
        m.insert(i, i);
    }
    assert!(m.len() == n);
    let m2 = m.split_off(&(n / 2));
    assert!(m2.len() == n / 2);
    assert!(m.len() + m2.len() == n);
}

fn bench_split_off(c: &mut Criterion) {
    let mut group = c.benchmark_group("SplitOff");
    for n in [10000].iter() {
        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| {
                exp_split_off_test(*n);
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| {
                std_split_off_test(*n);
            })
        });
    }
    group.finish();
}

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
