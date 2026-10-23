# Publishing to GitHub (Windows CMD, multiple accounts)

Target: **owner `vipintk2807`**, repository **`textile-defect-detection`**, profile name **Vipin T K**, SSH alias **`github-vipintk2807`** (if you have configured it). Nothing was created, authenticated or pushed by the build — you do all of this. Never paste passwords, tokens or private keys anywhere; these steps never need you to.

## 0. Extract and open the project
Extract `textile-defect-detection.zip` (right-click → Extract All), then in CMD:
```bat
cd /d C:\path\to\textile-defect-detection
```
Make sure the folder contains `README.md`, `app.py`, `src\` and **no** `.venv\`, dataset images, `*.pth` or `outputs\` contents you do not intend to publish.

## 1. Create the empty repository on GitHub (browser)
Sign in as **vipintk2807** → New repository → name `textile-defect-detection` → Public (or Private) → **do not** add a README/.gitignore/licence (the project already has them) → Create.

## 2. Check which identity and SSH setup apply
```bat
git --version
git config --global user.name
git config --global user.email
type "%USERPROFILE%\.ssh\config"
```
Look for a block like (the key filename is yours; do not post its contents):
```
Host github-vipintk2807
    HostName github.com
    User git
    IdentityFile ~/.ssh/<your key file for vipintk2807>
    IdentitiesOnly yes
```
If that alias does not exist, do not edit other accounts' entries; add only a new block for this account, or use HTTPS with GitHub's browser/credential-manager sign-in instead.

Test the alias authenticates as the right account:
```bat
ssh -T git@github-vipintk2807
```
Expected greeting: `Hi vipintk2807! You've successfully authenticated…`. If it greets a different username, **stop** — wrong key/alias.

## 3. Initialise Git with a repository-local identity
```bat
git init -b main
git config --local user.name "Vipin T K"
git config --local user.email "<your GitHub noreply or chosen email>"
git config --local user.name
git config --local user.email
```
(`--local` leaves your global/other-account settings untouched. GitHub's noreply address is under Settings → Emails.)

## 4. Review what will be committed
```bat
git add .
git status
git ls-files
```
Confirm the list contains only source, tests, configs, docs, notebook — **no** `data\raw`, `*.pth`, `*.db`, `outputs\…`, `.env`, or planning documents. If something unwanted appears: `git rm -r --cached <path>` and fix `.gitignore`.

## 5. Commit and connect the remote
```bat
git commit -m "Initial implementation of textile defect detection baseline"
git remote add origin git@github-vipintk2807:vipintk2807/textile-defect-detection.git
git remote -v
git branch --show-current
```
Check that the remote shows `github-vipintk2807:vipintk2807/textile-defect-detection.git` and the branch is `main`.

(HTTPS alternative: `git remote add origin https://github.com/vipintk2807/textile-defect-detection.git`.)

## 6. Push
```bat
git push -u origin main
```
Then open <https://github.com/vipintk2807/textile-defect-detection> and confirm it is under the right account.

## After publishing
* README claims must stay true: update `docs/VERIFICATION_REPORT.md` with your own test run before tagging a release.
* Do not commit datasets, weights or reports unless their licences allow it.
* Private planning documents belong in the separate `textile-defect-detection-planning` repository, not here.
