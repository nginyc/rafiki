function isEmail(value) {
  const expression = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+$/
  return expression.test(value)
} 


export function email(value) {
  return value && !isEmail(value.trim()) ? 'Invalid email' : null;
}

function isDirty(value) {
  return value || value === 0;
}

export function required(requiredFields, values) {
  return requiredFields.reduce(
    (fields, field) => ({
      ...fields,
      ...(isDirty(values[field]) ? undefined : { [field]: 'Required' }),
    }),
    {},
  );
}
