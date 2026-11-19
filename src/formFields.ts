// The fields of a form taken from a saved request or a saved draft: only the form's own fields.
// A saved request also holds what the server added to it (its kind, its sampling, a repair of another video).
// A form restored from a repaired video kept the repair, also in its saved draft, and sent it with the next
// video: that job failed at its start with "The original video does not match this repair".
export function formFields<T extends object>(form:T,saved:unknown):Partial<T>{
  if(!saved||typeof saved!=='object')return {};
  const values=saved as Record<string,unknown>;
  return Object.fromEntries(Object.keys(form).filter(key=>key in values).map(key=>[key,values[key]])) as Partial<T>;
}
