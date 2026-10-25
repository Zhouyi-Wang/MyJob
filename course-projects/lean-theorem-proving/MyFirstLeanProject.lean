import Mathlib

-- Problem 2.1: Factorial is Positive
theorem problem_2_1 :∀ n : ℕ , n! > 0 := by
  sorry

-- Problem 2.2: Factorial Growth
theorem problem_2_2:∀ (n : ℕ) , (h : 1 <= n) →  n! >= n := by
  sorry

-- Problem 2.3: Symmetry of Binomial Coefficients
theorem problem_2_3 : ∀ (n : ℕ) (k : ℕ) , (k <= n) → Nat.choose n k = Nat.choose n (n - k) := by
  sorry

-- Problem 2.4: Pascal's Identity — Statement Only
theorem problem_2_4 : ∀ (n : ℕ) (k : ℕ) , Nat.choose (n+1) (k+1) = Nat.choose n k +Nat.choose n (k+1) := by
  sorry

-- Problem 2.5: Pigeonhole Principle
theorem problem_2_5 : ∀ (n : ℕ) (f : Fin (n+1) → Fin n) , ¬ Function.Injective f := by
  sorry

-- Problem 2.6: Inclusion-Exclusion for Two Finite Sets
theorem problem_2_6 : ∀ {α : Type _} [DecidableEq α ](A B : Finset α), Finset.card (A ∪ B) + Finset.card (A ∩ B) = Finset.card A + Finset.card B := by
  sorry

-- Problem 2.7: Vandermonde's Identity
theorem problem_2_7 : ∀ (m n r: ℕ) , (Finset.range (r+1)).sum (fun k : ℕ ↦ Nat.choose m k * Nat.choose n (r-k) ) = Nat.choose (m+n) r := by
  sorry

-- Problem 2.8: Handshake Lemma — Statement
theorem problem_2_8(V : Type _) [Fintype V] (G : SimpleGraph V) [DecidableRel G.Adj] : (∑ v : V, G.degree v ) = 2 * G.edgeFinset.card  := by
  sorry

-- Problem 3.1: Gauss Sum
theorem gauss_sum (n : ℕ ) : 2 * ∑ i ∈ Finset.range (n + 1), i = n * (n + 1) := by
  induction n with
  | zero => norm_num
  | succ d ih =>
    rw [Finset.sum_range_succ, mul_add, ih]
    ring

-- Problem 3.2: Pascal's Identity
theorem pascal_identity (n k : ℕ ) :
 Nat.choose (n + 1) (k + 1) = Nat.choose n k + Nat.choose n (k + 1) := by
  induction n with
  | zero =>
    cases k with
    | zero =>norm_num
    | succ k' => simp [Nat.choose_zero_succ] <;> aesop
  |succ n ih =>
    cases k with
    |zero => simp [Nat.choose_succ_succ]
    |succ k' =>simp_all [Nat.choose_succ_succ]



-- Problem 3.3: Sum of binomial coefficients
theorem binom_row_sum (n : ℕ ) :
 ∑ k ∈  Finset.range (n + 1), Nat.choose n k = 2 ^ n := by
  exact Nat.sum_range_choose n

-- Problem 3.4: Pigeonhole Principle (constructive)
theorem pigeonhole (n : ℕ ) (f : Fin (n + 2) → Fin (n + 1)) :
 ∃ i j : Fin (n + 2), i ≠ j ∧ f i = f j := by
  have h_card : Fintype.card (Fin (n + 2)) > Fintype.card (Fin (n + 1)) := by
    have h1 : Fintype.card (Fin (n + 2)) = n + 2 := by
      exact Fintype.card_fin (n + 2)
    have h2 : Fintype.card (Fin (n + 1)) = n + 1 := by
      exact Fintype.card_fin (n + 1)
    rw [h1, h2]
    <;> omega  -- 证明 n + 2 > n + 1
  exact Fintype.exists_ne_map_eq_of_card_lt f h_card

-- Problem 3.5: Handshake Lemma
theorem handshake_lemma {V : Type*} [Fintype V] [DecidableEq V]
 (G : SimpleGraph V) [DecidableRel G.Adj] :
 ∑ v, G.degree v = 2 * G.edgeFinset.card := by
 exact SimpleGraph.sum_degrees_eq_twice_card_edges G
