import React from "react"
import { Form, Field } from 'react-final-form'
import { FormTextField, FormIntegerField, FormNumberField } from "mui-form-fields"

import { Button, Grid, ListItem, ListItemIcon, Icon, FormControl, InputLabel, Input, Select, MenuItem, FormHelperText } from '@material-ui/core'


function MyField(props) {

    const { name, render, icon } = props

    return (
        <ListItem style={{ paddingRight: '24px' }}>
            <ListItemIcon>
                <Icon>{icon}</Icon>
            </ListItemIcon>
            <Field
                name={name}
                render={({ input, meta }) => {
                    return render({ input, meta });
                }}
            />
        </ListItem>
    )
}

function MySelectField(props) {
    const { label, options, icon, name } = props

    return (
        <MyField icon={icon} name={name} render={({ input, meta }) => {
            return (
                < FormControl style={{ width: '100%' }}>
                    <InputLabel shrink>{label}</InputLabel>
                    <Select
                        value={input.value || ''}
                        onChange={input.onChange}
                        input={<Input name="train_dataset_it" />}
                    >
                        {options.map((o) => {
                            return (
                                <MenuItem key={o.value} value={o.value}>
                                    {o.label}
                                </MenuItem>
                            );
                        })}
                    </Select>
                </FormControl>
            )
        }
        }>
        </MyField>)
}

class CreateTrainJobForm extends React.Component {
    onSubmit = (values) => {
        alert("submitting", JSON.stringify(values, 0, 2))
    }

    render() {
        const options = [{ value: "A", label: "Some A" }, { value: "B", label: "Some B" }]

        return (
            <Form onSubmit={this.onSubmit}>
                {
                    ({ handleSubmit, invalid, values }) => {
                        return (
                            <Grid container spacing={3}>
                                <Grid item xs={12} lg={7}>
                                    <div style={{ textAlign: "center" }}>
                                        <FormTextField icon="chrome_reader_mode" name="name" label="Application Name" />
                                        <FormTextField icon="chrome_reader_mode" name="task" label="Task" placeholder="IMAGE_CLASSIFICATION" disabled />
                                        <MySelectField icon="perm_data_setting" name="train_dataset_id" label="Dataset for Traning" options={[
                                            { value: "A", label: "Some A" },
                                            { value: "B", label: "Some B" }
                                        ]} />
                                        <MySelectField icon="perm_data_setting" name="val_dataset_id" label="Dataset for Validation" options={[
                                            { value: "A", label: "Some A" },
                                            { value: "B", label: "Some B" }
                                        ]} />
                                        <FormIntegerField
                                            icon="extension"
                                            name="budget_gpus"
                                            label="Buget(GPUs)"
                                        />
                                        <FormNumberField
                                            icon="extension"
                                            name="budget_hours"
                                            label="Buget(Hours)"
                                        />
                                        <MySelectField icon="perm_data_setting" name="models" label="models" options={[
                                            { value: "A", label: "Some A" },
                                            { value: "B", label: "Some B" }
                                        ]} />
                                        <Button
                                            style={{ width: "200px" }}
                                            variant="contained" color="primary" disabled={invalid} onClick={(event) => {
                                                handleSubmit(event)
                                            }}>
                                            Submit
                                        </Button>
                                    </div>
                                </Grid>
                                <Grid item xs={12} lg={4}>
                                    <div style={{ background: "#ddd" }}>
                                        <p>{JSON.stringify(values, null, 2)}</p>
                                    </div>
                                </Grid>
                            </Grid>
                        )
                    }
                }
            </Form>
        )
    }

}

export default CreateTrainJobForm