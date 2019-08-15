import React from "react"
import { Form, Field } from 'react-final-form'
import { FormTextField, FormIntegerField, FormNumberField } from "mui-form-fields"

import { Button, Grid, ListItem, ListItemIcon, Icon, FormControl, InputLabel, Input, Select, MenuItem, Typography } from '@material-ui/core'


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
    const { label, options, icon, name, multi } = props

    return (
        <MyField icon={icon} name={name} render={({ input, meta }) => {
            return (
                < FormControl style={{ width: '100%' }}>
                    <InputLabel shrink>{label}</InputLabel>
                    <Select
                        onChange={input.onChange}
                        value={input.value || (multi ? [] : '')}
                        input={<Input name={name} />}
                        multiple={multi}
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
        const json = this._generateJSON(values)
        this.props.postCreateTrainJob(json)
    }

    _generateJSON(value) {
        return {
            "app": value.name,
            "task": "IMAGE_CLASSIFICATION",
            "train_dataset_id": value.train_dataset_id,
            "val_dataset_id": value.val_dataset_id,
            "budget": {
                "GPU_COUNT": parseInt(value.budgetGpus),
                "TIME_HOURS": parseFloat(value.budgetHours)
            },
            "model_ids":  value.models !== undefined ? value.models : [],
            "train_args": {}
        }
    }

    render() {
        // Options for datasets
        const options = this.props.datasets.map((dataset) => {
            return {
                value: dataset.id,
                label: dataset.name + "(" + dataset.id + ")"
            }
        })

        // Options for models
        const models = this.props.models.map((model) => {
            return {
                value: model.id,
                label: model.name + "(" + model.id + ")"
            }
        })

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
                                        <MySelectField icon="perm_data_setting" name="train_dataset_id" label="Dataset for Traning" options={options} />
                                        <MySelectField icon="perm_data_setting" name="val_dataset_id" label="Dataset for Validation" options={options} />
                                        <FormIntegerField
                                            icon="extension"
                                            name="budgetGpus"
                                            label="Buget(GPUs)"
                                        />
                                        <FormNumberField
                                            icon="extension"
                                            name="budgetHours"
                                            label="Buget(Hours)"
                                        />
                                        <MySelectField icon="perm_data_setting" name="models" label="models" multi={true} options={models} />
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
                                        <Typography>{JSON.stringify(this._generateJSON(values), null, 2) }</Typography>
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