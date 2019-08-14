export const Types = {
   // ==== TRAINJOBS ===== 
   REQUEST_TRAIN_JOBSLIST:"Jobs/request_train_jobslist", // async 
   POST_CREAT_TRAINJOB: "Jobs/post_create_trainjob", // async
   POPULATE_TRAINJOBSLIST: "Jobs/populate_trainjobslist", // sync
   REQUEST_STOP_TRAINJOB:"Jobs/request_stop_trainjob", // async

   // === TRIALS ====
   REQUEST_TRIALSLIST_OFJOB: "Trials/request_trialslist_ofjob",
   POPULATE_TRIALSTOJOBS: "Trials/populate_trialstojobs"
}

/* ====== JOBS ======= */

/* This action create new train job*/
export function createTrainJob(json) {
   return {
      type: Types.POST_CREAT_TRAINJOB,
      json
   }
}

/*=== request jobs list ===*/ 
/* This action is to request job list from remote server, would call
   dispatch action Types.POPULATE_TRAINJOBSLIST on success. */
export function requestJobsList() {
   return {
      type: Types.REQUEST_TRAIN_JOBSLIST
   }
}

/* This action is to update jobs list with the server, called when 
   receive respond from the server
*/
export function populateJobsList(jobsList) {
   return {
      type: Types.POPULATE_TRAINJOBSLIST,
      jobsList
   }
}

/* === stop train job === */
export function requestStopTrainJob(app, appVersion) {
   return {
      type: Types.REQUEST_STOP_TRAINJOB,
      app, appVersion
   }
}

/* ====== TRIALS ======= */

export function requestTrialsListOfJob(app, appVersion) {
   return {
      type: Types.REQUEST_TRIALSLIST_OFJOB,
      app, appVersion
   }
}

export function populateTrialsToJobs(trials, app, appVersion) {
   return {
      type: Types.POPULATE_TRIALSTOJOBS,
      trials,
      app,
      appVersion
   }
}