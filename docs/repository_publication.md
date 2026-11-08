# Publishing the reproducibility snapshot

1. Run `python scripts/reproduce.py verify` in the repository root. Commit
   the snapshot and push it to the new public GitHub repository.
2. Copy the published commit SHA and open the commit and
   `studies/sensitivity/` from a signed-out browser. The published commit
   should contain the S0, E1--E5 and Appendix H materials.
3. To use GitHub as the sole distribution route, update the main article's
   availability statement, the supplement's opening note and Appendix H's
   record references with the published commit and a direct link to its
   `studies/sensitivity/` directory. Rebuild both PDFs. The V15 PDFs refer
   to the named supplementary ZIP, so include that ZIP when submitting V15
   as compiled.

## Availability wording for a published repository snapshot

> The source code and synthetic data for S0, E1--E5 and the sensitivity
> analyses in Supplementary Appendix H are available in the cited repository
> at the cited commit. The repository includes the protocols, random seeds,
> code, numerical records and model checkpoints. The Appendix H materials
> are in the repository's `studies/sensitivity/` directory.

Insert the actual public commit and directory URLs when using this wording.
