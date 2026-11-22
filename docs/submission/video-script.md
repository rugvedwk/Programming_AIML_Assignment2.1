# Recording the NumCompute streaming demo

Aim to finish at 4:35. The screen cues allow time to run cells and point to results. Read only the quoted text. The final 25 seconds are spare.

Before recording, follow the installation instructions in the README on `assignment-2.2-rugved`. Open `demo/stream_demo.ipynb` in the installed environment and have `benchmark/stream_results.md` ready in another tab. Increase the notebook zoom until the log headings and labels are legible.

### 0:00–0:35 · Setup and execution

Show the active terminal environment and run the following. Switch to the notebook, choose the environment’s kernel and restart it with **Run All**.

```sh
git branch --show-current
python --version
python -m pip check
jupyter notebook demo/stream_demo.ipynb
```

> This is my individual streaming extension of NumCompute. I’m on the assignment branch. I installed the project with the development extra for Jupyter and pytest. The package uses NumPy for computation and Matplotlib for plotting. I’ll run the notebook from a fresh kernel and follow one forecast from its CSV row through to its evaluation.

### 0:35–1:20 · Data and pipeline

Show the feature list, then `weather_chunks`, `make_trainer` and the `fit_chunk` call in **Replay the CSV**.

> The group package’s CSV reader gives us ten Adelaide observations at a time in date order. RainTomorrow is the answer we want to predict. Today’s rainfall and humidity are among the seven inputs. The date keeps the rows in order. The pipeline fills missing values, scales the features and fits a depth-five decision tree. For every chunk, the trainer first predicts using the currently deployed tree, then passes the observations and labels to partial fit. This keeps tomorrow’s answer out of today’s forecast.

### 1:20–2:05 · First retrain and first scored chunk

Under **Inspect warm-up and retraining**, point to the Pending, Fit and Chunk accuracy columns for chunks 1 to 4, then the time and memory columns.

> With a buffer of thirty, the pending count moves from ten to twenty. The third chunk takes it to thirty. A new tree trains on all thirty rows and the count drops to zero. That third chunk has no accuracy because there was no tree when the chunk arrived. The fourth chunk is the first one we can score. Each retrain freezes the current preprocessing with the tree, so a change in the running mean cannot move an old decision threshold. The log also shows elapsed update time and retained array storage.

### 2:05–2:55 · Accuracy and fair comparisons

Show the per-chunk and rolling plots, then **Compare buffer sizes on common rows** and its two-model plot. Point to chunk 13.

> The first line shows accuracy on each ten-day chunk. The second smooths it over the last forty scored days. The gap at the beginning represents warm-up. A larger buffer waits longer before it can predict, so the table compares all three models on the same final two hundred and forty days. The thirty-row tree gets about seventy-one percent right and retrains twelve times. The largest gets about sixty-six percent and retrains three times. Predicting no rain every day gets about seventy-four percent here, which puts those results in perspective.

### 2:55–3:30 · Inspect individual forecasts

Show **Look at the latest predictions**. Point to 24 and 25 December in the table and to their corresponding positions in the plot.

> Here are the final ten forecasts saved before training. On December twenty-fourth the model forecast rain for the next day and the observed label was no. On the twenty-fifth it forecast rain and the label was yes. Looking at these pairs shows who pays the cost of a false alarm or a missed warning. The overall accuracy alone hides that distinction.

### 3:30–4:35 · Split speed and response to change

Switch to `benchmark/stream_results.md`. Show the 10,000-row split result, then the before-drift, early-after-drift and retrain columns. End on the report’s drift figure if time allows.

> The split benchmark checks that the Python and NumPy versions pick the same feature, threshold and missing-value route before timing them. At ten thousand rows the vectorised root search was about eight times faster. A separate experiment changes the correct label rule halfway through an identical stream for all three models. In the first thousand and twenty-four rows after that change, the small buffer gets about seventy-six percent right and the largest about ten percent. The smaller buffer adapts sooner, but it retrains thirty-two times over the run against four for the largest. My choice would depend on the cost of stale predictions and the available training time.

Stop the recording and check that the exported MP4 or shareable recording is at most five minutes.
