import React from "react"
import { bindActionCreators, compose } from 'redux'
import PropTypes from 'prop-types';
import { connect } from "react-redux"
import * as actions from "containers/Datasets/actions"

import { Button, Grid } from "@material-ui/core"
import { makeStyles } from "@material-ui/styles"

// Third part dependencies
import { Form, Field } from "react-final-form"
import { useDropzone } from "react-dropzone";
import { FormTextField, FormSwitchField } from "mui-form-fields"

const useStyles = makeStyles({
    dropzoneWrapper: {
        border: "1px solid black",
        borderRadius: 1,
        padding: "5px",
        margin: "10px",
        textAlign: "center"
    }
})

function DatasetsDropzone(props) {
    const classes = useStyles()
    const { getRootProps, getInputProps, acceptedFiles, isDragActive } = useDropzone()

    return (
        <Field name="dataset" lable="Upload Datasets from your Computer">
            {
                ({ input, meta }) => {
                    React.useEffect(
                        () => {
                            input.onChange(acceptedFiles)
                        }
                        , [acceptedFiles])
                    return (
                        <div className={classes.dropzoneWrapper} {...getRootProps()}>
                            <input {...getInputProps()}>
                            </input>
                            {isDragActive ? (
                                <p>Drop the files here ...</p>
                            ) : (
                                    <p>Drag 'n' drop some files here, or click to select files</p>
                                )}
                            <Button>Click to Upload</Button>
                        </div>
                    )
                }
            }
        </Field>
    )
}

class UploadDatasetsForm extends React.Component {
    static propTypes = {
        // classes: PropTypes.object.isRequired,
        postCreateDataset: PropTypes.func,
    }

    state = {
        fromLocal: false
    }

    onSubmit = (values) => {
        console.log("submit values", values)
        // Dispatch actions 
        if (this.state.fromLocal) {
            this.props.postCreateDataset(values.name, "IMAGE_CLASSIFICATION", values.dataset[0])
        } else {
            this.props.postCreateDataset(values.name, "IMAGE_CLASSIFICATION", undefined, values.dataset_url)
        }
    }

    render() {
        return (
            <div style={{ textAlign: "center" }}>
                <Form onSubmit={this.onSubmit}>
                    {
                        ({ handleSubmit, values, invalid }) => {
                            return (
                                <React.Fragment>
                                    <Grid container spacing={3}>
                                        <Grid item xs={12} lg={7}>
                                            <FormTextField icon="chrome_reader_mode" name="name" label="Dataset Name" />
                                            <FormSwitchField
                                                icon="attach_file"
                                                onClick={event => {
                                                    const value = Boolean(event.target.checked);
                                                    this.setState({ fromLocal: value });
                                                }}
                                                name="fromLocal"
                                                label="Upload from Computer"
                                            />
                                            {this.state.fromLocal ? <DatasetsDropzone /> :
                                                <FormTextField icon="cloud_upload" name="dataset_url" label="Upload Datasets From url" />}
                                            <Button
                                                style={{ width: "200px" }}
                                                variant="contained" color="primary" disabled={invalid} onClick={(event) => {
                                                    handleSubmit(event)
                                                }}>
                                                Submit
                                            </Button>
                                        </Grid>
                                        <Grid item xs={12} lg={4}>
                                            <div style={{ background: "#ddd" }}>
                                                <p>{JSON.stringify(values, null, 2)}</p>
                                            </div>
                                        </Grid>
                                    </Grid>
                                </React.Fragment>
                            )
                        }
                    }
                </Form>
            </div>
        )
    }
}

function mapDispatchToProps(dispatch) {
    return bindActionCreators({ postCreateDataset: actions.postCreateDataset }, dispatch)
}

export default compose(
    connect(null, mapDispatchToProps)
)(UploadDatasetsForm);
