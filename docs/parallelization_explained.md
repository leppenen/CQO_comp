# Lecture Explanation: Parallelization in `ex_wqed_cluster.py`

## The Problem
We need to solve 20 different ODE systems (one for each `r` value in the parameter sweep).  
**Without parallelization:** solve them one at a time sequentially (slow).  
**With parallelization:** solve multiple at the same time on different CPU cores (fast).

---

## Part 1: `solve_system_wrapper`

```python
def solve_system_wrapper(args):
    """Wrapper for parallelization."""
    return solve_system_prop(*args)
```

### What it does
It's a **simple "unpacking" function**. It takes a single tuple `args` and passes its contents to `solve_system_prop`.

### Why we need it
The multiprocessing pool can only pass **one argument** to each worker. But `solve_system_prop` needs 7 arguments:
```
N, gamma, Dnm, Gnm, Omega_n, t_span, y0
```

So we:
1. **Pack** these 7 arguments into one tuple: `(N, gamma, Dnm, Gnm, Omega_n, t_span, y0)`
2. Pass that tuple to the pool
3. `wrapper` **unpacks** it with `*args` and calls `solve_system_prop`

**Analogy:** imagine you need to mail a package to 10 people. The mailbox only accepts sealed packages (one argument). So you pack all items into a box, send it, and the recipient unpacks it.

---

## Part 2: `parallel_solve_omega`

```python
def parallel_solve_omega(N, gamma, Dnm, Gnm, Omega_n_values, t_span, y0, n_workers=4):
    """Parallelized execution for solving ODEs across different Omega values."""
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(
                solve_system_wrapper,
                [(N, gamma, Dnm, Gnm, Omega_n, t_span, y0) for Omega_n in Omega_n_values],
                chunksize=2
            ),
            total=len(Omega_n_values),
            desc="Solving ODE sweep"
        ))
    return results
```

### Breaking it down step by step:

#### Line 1: Create a worker pool
```python
with mp.Pool(processes=n_workers) as pool:
```
- Creates `n_workers` independent Python processes (default: 4 workers)
- Each worker can solve one ODE system simultaneously
- `with` ensures cleanup when done

#### Line 2-7: Create a list of packed jobs
```python
[(N, gamma, Dnm, Gnm, Omega_n, t_span, y0) for Omega_n in Omega_n_values]
```
- For each `Omega_n` value in our sweep, create one tuple with all arguments
- 20 Omega values → 20 tuples = 20 independent jobs

#### Line 3: Send jobs to workers
```python
pool.imap(solve_system_wrapper, [...], chunksize=2)
```
- `pool.imap`: map (apply) the wrapper function to each tuple
- `chunksize=2`: give each worker 2 jobs at a time (balance efficiency & overhead)
- Returns results in **order** (important!)

#### Line 4-7: Show progress with tqdm
```python
tqdm(..., total=len(Omega_n_values), desc="Solving ODE sweep")
```
- Wraps the result generator to show a progress bar
- `total=...`: tell tqdm how many jobs there are
- `desc=...`: label for the progress bar

#### Line 8: Convert to list
```python
results = list(...)
```
- Collects all results into a Python list

---

## Visual Flow

```
main() calls:
  parallel_solve_omega(nz, gammap, Dnm, Gnm, [Omega_1, Omega_2, ..., Omega_20], ...)

Inside:
  Create Pool with 4 workers
  
  Job queue: [(N, gamma, ..., Omega_1, ...), 
              (N, gamma, ..., Omega_2, ...), 
              ..., 
              (N, gamma, ..., Omega_20, ...)]
  
  Distribute to workers:
  
    Worker 1    Worker 2    Worker 3    Worker 4
    Job 1       Job 2       Job 3       Job 4
    (solving)   (solving)   (solving)   (solving)
    
    ✓ done      ✓ done      ✓ done      ✓ done
    
    Job 5       Job 6       Job 7       Job 8
    (solving)   (solving)   (solving)   (solving)
    ...
```

## Timeline Example

**Without parallelization** (sequential):
- Job 1: 2 sec → Job 2: 2 sec → Job 3: 2 sec → ... → Job 20: 2 sec
- **Total: 40 seconds**

**With parallelization** (4 workers):
- Jobs 1–4 run at same time: 2 sec
- Jobs 5–8 run at same time: 2 sec
- Jobs 9–12 run at same time: 2 sec
- Jobs 13–16 run at same time: 2 sec
- Jobs 17–20 run at same time: 2 sec
- **Total: ~10 seconds** (5× speedup with 4 workers!)

---

## Key Takeaway

Parallelization is useful when:
- You have many **independent** jobs (different Omega values don't depend on each other)
- Each job takes significant time (ODE solving: ~2 sec per job)
- You have multiple CPU cores available

On Chemfarm with 10–20 CPU cores per node, you can get 10–20× speedup compared to sequential!
