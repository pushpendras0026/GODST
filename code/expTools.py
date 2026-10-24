# expTools.py
# Small helpers that every experiment uses: result folders, log file, saving tables,
# and picking datasets / values (normal run, --quick run, or what the user typed).

import os
import datetime
import pickle
import pandas as pd

projectFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def makeResultFolder(folderName):
    path = os.path.join(projectFolder, "results", folderName)
    os.makedirs(path, exist_ok=True)
    return path


def startLog(folder, stageName, settings):
    logPath = os.path.join(folder, "log.txt")
    with open(logPath, "w") as f:
        f.write("stage: " + stageName + "\n")
        f.write("started: " + str(datetime.datetime.now()) + "\n")
        f.write("settings: " + str(settings) + "\n\n")
    print("writing results to", folder)


def log(folder, text):
    print(text, flush=True)
    with open(os.path.join(folder, "log.txt"), "a") as f:
        f.write(text + "\n")


def saveTable(rows, folder, fileName="results.csv"):
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(folder, fileName), index=False)
    return table


def saveText(text, folder, fileName):
    with open(os.path.join(folder, fileName), "w") as f:
        f.write(text)


def pickList(normalList, quickList, settings, key=None):
    # if the user gave a value on the command line we only use that one
    if key is not None and settings.get(key) is not None:
        return [settings[key]]
    if settings.get("quick"):
        return quickList
    return normalList


def pickTimeLimit(normalLimit, quickLimit, settings):
    if settings.get("timelimit") is not None:
        return settings["timelimit"]
    if settings.get("quick"):
        return quickLimit
    return normalLimit


def roundOrNone(value, digits=6):
    if value is None:
        return None
    return round(value, digits)


def pickJobs(settings):
    # how many runs at the same time (--jobs). Default: all cores but two
    if settings.get("jobs") is not None:
        return max(1, settings["jobs"])
    cores = os.cpu_count()
    if cores is None:
        cores = 2
    return max(1, cores - 2)


def jobFileName(job):
    # a file name made from the job, for example ('car', 0, 'gosdt') -> car_0_gosdt.pkl
    text = str(job)
    if len(text) > 150:
        # long jobs (with a list of settings inside) get a short name
        import hashlib
        text = text[:100] + "_" + hashlib.md5(text.encode()).hexdigest()[:10]
    safe = ""
    for ch in text:
        if ch.isalnum() or ch in "._-":
            safe = safe + ch
        else:
            safe = safe + "_"
    return safe + ".pkl"


def runAndSave(item):
    # runs one job in a worker process and saves its answer to a file right away
    jobFunction, job, path = item
    answer = jobFunction(job)
    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(answer, f)
    return answer


def runJobs(jobFunction, jobList, numJobs, folder=None, checkpointFolder=None):
    # runs jobFunction(job) for every job, numJobs of them at the same time.
    # jobFunction must be a normal function at the top of a file (so the other processes find it).
    # Every finished job is saved in <folder>/checkpoints/ (or in checkpointFolder if given, so
    # different stages can share jobs), and if we start again, the jobs that are already there
    # are not run again. So a long run that stops can just be started again.
    checkFolder = None
    if checkpointFolder is not None:
        checkFolder = checkpointFolder
    elif folder is not None:
        checkFolder = os.path.join(folder, "checkpoints")
    if checkFolder is not None:
        os.makedirs(checkFolder, exist_ok=True)
    answers = []
    todo = []
    for job in jobList:
        path = None
        if checkFolder is not None:
            path = os.path.join(checkFolder, jobFileName(job))
            if os.path.exists(path):
                with open(path, "rb") as f:
                    answers.append(pickle.load(f))
                continue
        todo.append((jobFunction, job, path))
    if folder is not None and len(answers) > 0:
        log(folder, str(len(answers)) + " jobs were already done before, running the other " + str(len(todo)))

    if numJobs <= 1:
        for item in todo:
            answers.append(runAndSave(item))
            if folder is not None:
                log(folder, "finished job " + str(len(answers)) + "/" + str(len(jobList)) + ": " + str(item[1]))
        return answers
    import multiprocessing
    # maxtasksperchild=1: every job gets a fresh process, so a big job can not leave its
    # memory behind for the next job (and the time of one job does not depend on the one before)
    pool = multiprocessing.Pool(numJobs, maxtasksperchild=1)
    for answer in pool.imap_unordered(runAndSave, todo):
        answers.append(answer)
        if folder is not None:
            log(folder, "finished job " + str(len(answers)) + "/" + str(len(jobList)))
    pool.close()
    pool.join()
    return answers
