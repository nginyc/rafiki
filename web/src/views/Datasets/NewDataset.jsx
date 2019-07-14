/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

import React from "react";
import { Link } from "react-router-dom";

// @material-ui/core components
import withStyles from "@material-ui/core/styles/withStyles";
import { Icon, CircularProgress } from "@material-ui/core";

import TextField from '@material-ui/core/TextField';

// core components
import GridItem from "components/Grid/GridItem.jsx";
import GridContainer from "components/Grid/GridContainer.jsx";
import CustomInput from "components/CustomInput/CustomInput.jsx";
import Button from "components/CustomButtons/Button.jsx";
import Card from "components/Card/Card.jsx";
import CardHeader from "components/Card/CardHeader.jsx";
import CardBody from "components/Card/CardBody.jsx";
import CardFooter from "components/Card/CardFooter.jsx";
import FormHelperText from '@material-ui/core/FormHelperText';
import { findAllByDisplayValue } from "@testing-library/react";

const styles = {
  cardCategoryWhite: {
    color: "rgba(255,255,255,.62)",
    margin: "0",
    fontSize: "14px",
    marginTop: "0",
    marginBottom: "0"
  },
  cardTitleWhite: {
    color: "#FFFFFF",
    marginTop: "0px",
    minHeight: "auto",
    fontWeight: "300",
    fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    marginBottom: "3px",
    textDecoration: "none"
  },
  uploadUrl: {
  }
};

class NewDataset extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      name: "",
      task: "IMAGE_CLASSIFICATION",
      datasets_url: "",
      "submit_status": "none" // "none", "submitting", "success", "failed"
    }
    this.handleInputChange = this.handleInputChange.bind(this)
    this.handleClickButton = this.handleClickButton.bind(this)
  }

  handleInputChange = (event) => {
    const target = event.target;
    const value = target.type === "file" ? target.files[0] : target.value;
    const name = target.name;
    this.setState({
      [name]: value
    });
  }

  async handleClickButton(event) {
    event.preventDefault();
    const { appUtils: { rafikiClient, showError } } = this.props;

    try {
      console.log("button clicked")
      this.setState({ "submit_status":"submitting" })
      const {name, task, file, dataset_url } = this.state
      await rafikiClient.createDataset(name,task,file,dataset_url)
      alert("Successfully created the dataset!")
      this.setState({ "submit_status":"succeed" })
      this.props.history.push('/admin/datasets')
      debugger
    } catch (error) {
      this.setState({ "submit_status": "failed" })
      showError(error, 'Failed to create the dataset');
    }
  }

  render() {
    const { classes } = this.props;

    let SubmitButton = null
    
    switch (this.state.submit_status) {
      case "submitting":
        SubmitButton = <CircularProgress />
        break;
      case "succeed":
        SubmitButton = (<Link to="/admin/datasets">
          <Button color="primary">Back</Button>
        </Link>)
        break;
      default:
        SubmitButton = (
          <Button onClick={this.handleClickButton} color="primary">CREATE NEW DATASET</Button>
        )
      break;
    }

    return (
      <div>
        <GridContainer justify="center" alignContent="center">
          <GridItem xs={12} sm={12} md={12} lg={12} xl={12}>
            <Card>
              <CardHeader color="info">
                <h4 className={classes.cardTitleWhite}><Icon>add_photo_alternate</Icon>New Dataset</h4>
                <p className={classes.cardCategoryWhite}>Image Classification</p>
              </CardHeader>
              <CardBody>
                <GridContainer justify="center" alignContent="center">
                  <GridItem xs={12} sm={12} md={12} lg={12} xl={12}>
                    <TextField
                      name="name"
                      id="name"
                      label="Name"
                      placeholder="fashion_mnist"
                      helperText="Dataset Name"
                      fullWidth
                      margin="normal"
                      onChange={this.handleInputChange}
                    />
                    <TextField
                      name="task"
                      id="standard-full-width"
                      label="Task"
                      placeholder="IMAGE_CLASSIFICATION"
                      defaultValue="IMAGE_CLASSIFICATION"
                      helperText="Task Type"
                      fullWidth
                      disabled
                      margin="normal"
                      onChange={this.handleInputChange}
                    />
                  </GridItem>
                  <GridItem xs={12} sm={12} md={12}>
                    <h4 className={classes.cardCategory}>Upload from your computer</h4>
                    <FormHelperText id="component-helper-text"><p>*The dataset file must be of the <b>.zip archive</b> format with a <b>images.csv </b>at the root of the directory. The <b>images.csv</b> should be of a <b>.CSV</b> format with 2 columns of <b>path and class</b>. Please refer to more details <a href="https://nginyc.github.io/rafiki/docs/latest/src/user/datasets.html#dataset-type-image-files">here</a></p></FormHelperText>
                    <input
                      accept="zip/*"
                      name="file"
                      id="contained-button-file"
                      multiple
                      type="file"
                      onChange={this.handleInputChange}
                    />
                    <label htmlFor="contained-button-file">
                      <Button variant="contained" component="span" className={classes.button}>
                        Upload
                      </Button>
                    </label>
                    {' '} or {' '}
                    <TextField
                      className={classes.uploadUrl}
                      name="dataset_url"
                      label="Upload dataset from a URL"
                      placeholder=""
                      margin="none"
                      variant="outlined"
                      onChange= {this.handleInputChange}
                    />
                  </GridItem>
                </GridContainer>
              </CardBody>
              <CardFooter>
                { SubmitButton }
              </CardFooter>
            </Card>
          </GridItem>
        </GridContainer>
      </div>
    );
  }
}

export default withStyles(styles)(NewDataset);
