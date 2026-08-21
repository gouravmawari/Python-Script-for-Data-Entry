Task 4 - data Issues Found

found these by actually loading all 3 CSVs into pandas and checking real people, not just skimming.

1.no common ID across files ource3 has no email only phone fixed by matching on email first phone as fallback

2.one key wasnt enough — same person linked via email in one file and phone in another but a single key match m2issed the connection fixed with union-find so matches chain across sources

3.phone numbers in 4+ formats with/without +91, spaces, dashes. Normalized all to plain 10-digit

4.phones turned into scientific notation pandas auto-converted to float during merge Forced string dtype everywhere

5.one rows columns were shifted by one position email, name, skills etc all rotated Detected and un-rotated it

6.header row duplicated as a data row in source3 filtered it out

7.inconsistent casing — city status verified fields all mixed case normalized everything

8.CTC values off by ~100000x for some rows looks like lakhs vs rupees mix-up didnt guess-fix just flagged as suspect

9.duplicate rows with a name variant "R. Verma" vs "Rohit Verma" same person kept the fuller name

10.same namedifferent phone no way to know if same person didnt auto-merge flagged into a review queue instead

11.blank phones broke the DB insert went in as nan instead of null crashed on the unique constraint fixed the dtype cast before insert


## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python task1.py
This creates:
- merged_people.csv
- review_queue.csv
To upload merged_people.csv to Supabase, create a .env file:
DATABASE_URL=your_supabase_pooler_connection_string
Then run:
python task1DB.py