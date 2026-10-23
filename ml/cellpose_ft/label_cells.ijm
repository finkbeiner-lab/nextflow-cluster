// label_cells.ijm -- SNT-style ground truth for Cellpose fine-tuning (miniSOG-RGEDI GFP16).
// Walks through every crop in a labeling folder made by make_fiji_set.py: opens the crop,
// loads the model's outlines into the ROI Manager, lets you fix them, and saves
// <tag>_edited.zip. Crops that already have an _edited.zip are skipped, so you can stop
// anytime and pick up later. To redo a crop, delete its _edited.zip and re-run.
// Run: Plugins > Macros > Run... > label_cells.ijm

dir = getDirectory("Choose the labeling folder (contains manifest.csv)");
if (!File.exists(dir + "manifest.csv")) exit("No manifest.csv in " + dir);
lines = split(File.openAsString(dir + "manifest.csv"), "\n");
header = split(replace(lines[0], "\r", ""), ",");
ti = -1; ki = -1; si = -1; li = -1; di = -1; pi = -1; ri = -1;
for (i = 0; i < header.length; i++) {
    if (header[i] == "tag") ti = i;
    if (header[i] == "kind") ki = i;
    if (header[i] == "split") si = i;
    if (header[i] == "line") li = i;
    if (header[i] == "dose") di = i;
    if (header[i] == "tp") pi = i;
    if (header[i] == "round") ri = i;
}
n = lines.length - 1;
done = 0;
for (k = 1; k <= n; k++) {
    f = split(replace(lines[k], "\r", ""), ",");
    if (File.exists(dir + f[ti] + "_edited.zip") || File.exists(dir + f[ti] + "_edited_EMPTY.txt")) done++;
}

help = "Outlines = the model's guess. Make every cell in this crop right:\n \n"
     + "SELECT a cell: Rectangle tool, click inside its outline.\n"
     + "GROW / add a tail: Selection Brush (right-click the Oval tool),\n"
     + "      shift-drag to add, alt-drag to remove -> ROI Manager: Update\n"
     + "DELETE junk / debris: select it -> ROI Manager: Delete\n"
     + "NEW cell: Freehand tool, trace the outline -> press T\n"
     + "SPLIT a merged pair: delete it, trace each cell -> T\n"
     + "MERGE a split cell: delete the pieces, trace the whole cell -> T\n \n"
     + "Rule: label every cell-sized object, including round/compact dying cells.\n"
     + "Skip puncta much smaller than a nucleus. Trace dim tails to where GFP ends.\n"
     + "Contrast: Shift+C (B&C) -- display only, never changes the data.\n \n"
     + "Click OK when this crop is finished (it saves automatically).";

for (k = 1; k <= n; k++) {
    f = split(replace(lines[k], "\r", ""), ",");
    tag = f[ti];
    out = dir + tag + "_edited.zip";
    if (File.exists(out) || File.exists(dir + tag + "_edited_EMPTY.txt")) continue;

    open(dir + tag + ".tif");
    img = getTitle();
    run("Enhance Contrast", "saturated=0.5");
    roiManager("reset");
    roiManager("Open", dir + tag + "_prefill.zip");
    roiManager("Show All without labels");
    call("ij.gui.Toolbar.setBrushSize", "6");
    setTool("rectangle");

    waitForUser("Crop " + (done + 1) + " of " + n + ": " + tag + "  [" + f[li] + " " + f[di] + " " + f[pi]
                + ", " + f[ki] + ", " + f[si] + "]", help);

    // commit a pending edit that was never added/updated (common slip)
    selectWindow(img);
    if (selectionType() != -1) {
        idx = roiManager("index");
        if (idx >= 0) roiManager("update"); else roiManager("add");
    }
    roiManager("Deselect");
    run("Select None");
    cnt = roiManager("count");
    if (cnt > 0) roiManager("Save", out);
    else File.saveString("no cells", dir + tag + "_edited_EMPTY.txt");
    close(img);
    roiManager("reset");
    done++;
    if (ri >= 0 && k < n) {
        nf = split(replace(lines[k + 1], "\r", ""), ",");
        if (f[ri] == "1" && nf[ri] == "2") {
            if (!getBoolean("Round-1 set complete (" + done + " crops).\n \nThat's enough for the first training run -- tell Claude.\nThe tuned model will prefill the remaining crops, which makes them faster.\n \nKeep going with the remaining crops anyway?", "Keep going", "Stop here"))
                exit("Round 1 done. Re-run the macro later to label the rest.");
            continue;
        }
    }
    if (done < n) {
        if (!getBoolean("Saved " + tag + " (" + cnt + " cells).  " + done + " of " + n + " done.\n \nContinue to the next crop?",
                        "Next crop", "Stop for now")) exit("Stopped. Re-run the macro to pick up where you left off.");
    }
}
showMessage("All " + n + " crops labeled", "All crops have _edited.zip files.\nTell Claude they're ready.");
