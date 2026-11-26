# Progress Report 2: short script

About 12 minutes. Slides come from `Denguide_Progress_Deck_Check2.pptx` (22 slides) and
`Denguide_New_Slides_Code_and_Paper.pptx` (7 slides, marked **New 1** to **New 7**).
The speaker split is a suggestion; every member should be ready to explain any slide.

---

## Part 1: the system (Ivan)

**1. Title**
Good day. We are Group 11, and this is our second progress report on Denguide, a chatbot that answers questions using the DOH dengue guideline.

**2. The problem**
The DOH dengue guideline has 211 pages and nine clinical questions. A doctor cannot search it during a consult. Denguide finds the right part, answers with the page number, and says so when the guideline has no answer.

**3. How Denguide answers a question**
We clean the guideline once, cut it into pieces, and save them in a database. For each question, Denguide finds the four closest pieces. If the best one scores 0.61 or more, the AI answers with page numbers. If not, Denguide says the topic is not covered and does not ask the AI.

**4. Four words we use**
Four words come up often. A piece is a short part of the guideline. The match score shows how close a piece is to the question. The answer rules are the instructions the AI follows. Practice questions are for improving the system, and the final questions stay locked until the end.

**5. How we cleaned the guideline**
We kept the main 114 pages, read the printed page number on every page, and removed citations, grading symbols and reference lists. This left about 195,000 characters of clean text.

**6. How we cleaned the tables**
Plain copying puts one value per line, so the AI cannot tell which number is which. We read each cell by its position and keep every row with its labels. We rebuilt 53 tables this way.

**7. Cutting the guideline into pieces**
We cut the text into 151 pieces of up to 350 words. Pieces follow the guideline's own questions and sections, and the last 50 words repeat in the next piece so no idea is cut in half.

**New 1. Code: cleaning the text**
This is the actual cleaning code. It deletes citation numbers and grading symbols, fixes spaces, rejoins words split across lines, and reads the printed page number.

**New 2. Code: rebuilding the tables**
Here each table cell is read by its position on the page. Every row becomes "Header: value", and a table split across two pages keeps its header.

**New 3. Code: cutting into pieces**
These are the settings: 350 words per piece, 50 words of overlap, and pieces under 12 words dropped. A new piece also starts whenever the section changes.

**8. The models we use**
BGE-M3 finds the closest pieces. We compare two AI models: Qwen3-8B, the main one, and Gemma 3 12B. Both are free and run on Google Colab's free GPU.

**New 4. Code: search and the 0.61 line**
This code checks only the best piece against 0.61. Below the line, Denguide gives the fixed reply. We chose 0.61 as the middle of the best range on practice questions.

**New 6. Code: the answer rules (prompt v2)**
These are the nine rules the AI follows, version 2. The main ones: use only the pieces, cite a page after every sentence, and keep the guideline's own wording. Jessica reviewed and approved them.

**9. How we measure Denguide**
We measure seven things, from "right page found" to speed. The final test has 50 questions that were never used for tuning, and it runs only once.

**New 5. Code: checking the cited pages**
This code checks every answer. Each cited page must be one of the pieces the AI was given. This is how we count citation mistakes.

---

## Part 2: the runs (Ricci)

**10. All runs at a glance**
We recorded every run, including the failed ones. Three runs worked, two failed, and we found and fixed both causes. Run 06 is next.

**11. Run 01: build the search database**
We built the database and tested the search with 15 practice questions. The right page was always among the four pieces found, so we set the line at 0.61.

**12. Run 02: first AI answers (failed)**
Gemma gave blank answers, and one Qwen answer missed a fact. We switched Gemma to a more precise number setting and changed the line to check only the best piece.

**13. Run 03: both AI models, after the fixes**
Both models answered all 12 real questions and refused all 3 off-topic ones. Some answers still said "we recommend" or had sentences without a page, so we wrote stricter rules.

**14. Run 04: stricter answer rules**
Only the rules changed. "We recommend" disappeared, and sentences without a page dropped from 12 to 4 for Qwen and from 7 to 2 for Gemma. Taglish questions still got English answers.

**15. Run 05: 40 practice questions (failed)**
We added 25 practice questions. Qwen refused 10 of 11 off-topic ones, but Gemma ran out of GPU memory. We now move the search model off the GPU before the AI loads.

**16. Run 06: fresh start on 40 questions**
Run 06 is next. It becomes the starting point, and after it we test one improvement at a time.

---

## Part 3: examples and next steps (Jessica)

**17. Sample run: a normal question**
For "Should ORS be given?", both models follow the guideline and cite page 71. Qwen also gives the certainty of evidence. Gemma leaves it out.

**18. Sample run: a question the guideline does not cover**
For a paracetamol dose question, the closest piece scored 0.56, below the line. Denguide gave the fixed reply in 0.04 seconds without asking the AI.

**19. Sample run: a mistake we found**
For a Taglish question on lagundi and kamote tops, Qwen talked about papaya instead of refusing. It also said there were no tawa-tawa studies, which is wrong. We logged it and will fix it before the final test.

**New 7. IEEE paper progress**
The paper is about half done, with 2,450 words drafted. Methodology and the practice results are mostly written. The final results and conclusions come after the final test.

**20. Where we are now**
Search, refusals and page citations work. Taglish answers, a few off-topic questions and Gemma's speed still need work.

**21. What comes next**
Next are Run 06, small improvements, the final test, and then the paper, web app and video. We will consult before January 1, submit on January 8, and demo on January 11 to 13.

**22. Thank you**
Thank you. We welcome your questions and feedback.
