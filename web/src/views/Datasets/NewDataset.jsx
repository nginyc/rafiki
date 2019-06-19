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
      alert("Create Dataset Succeed")
      this.setState({ "submit_status":"succeed" })
    } catch (error) {
      this.setState({ "submit_status": "failed" })
      showError(error, 'Failed createDatabase');
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
          <Button onClick={this.handleClickButton} color="primary">CREATE NEW DATASETS</Button>
        )
      break;
    }

    return (
      <div>
        <GridContainer justify="center" alignContent="center">
          <GridItem xs={12} sm={12} md={12} lg={12} xl={12}>
            <Card>
              <CardHeader color="info">
                <h4 className={classes.cardTitleWhite}><Icon>add_photo_alternate</Icon>New Datasets</h4>
                <p className={classes.cardCategoryWhite}>Image Claasification</p>
              </CardHeader>
              <CardBody>
                <GridContainer justify="center" alignContent="center">
                  <GridItem xs={12} sm={12} md={12} lg={12} xl={12}>
                    <TextField
                      name="name"
                      id="name"
                      label="Name"
                      placeholder="fashion_mnist_app"
                      helperText="Datasets Name"
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
                    <FormHelperText id="component-helper-text"><p>*The datasets file must be in <b>.zip</b> archive format. inside which each sub-folder corresponds to a category. The train & test dataset's images should have the <b>same dimensions W x H</b>, accepted formats are  <b>JPEG or PNG</b></p></FormHelperText>
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
                      label="Datasets from a url"
                      placeholder=""
                      margin="none"
                      variant="outlined"
                      onChange= {this.handleInputChange}
                    />
                  </GridItem>
                  <GridItem xs={12} sm={12} md={12}>
                    <h6 className={classes.cardCategory}>Data Distributuion</h6>
                    <FormHelperText id="component-helper-text">*On Traning, Rafiki system will set aside 5% for validation</FormHelperText>
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
