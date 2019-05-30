import React from "react";
import { Link } from "react-router-dom";
// @material-ui/core components
import withStyles from "@material-ui/core/styles/withStyles";
import Icon from "@material-ui/core/Icon";
import TextField from '@material-ui/core/TextField';

// core components
import GridItem from "components/Grid/GridItem.jsx";
import GridContainer from "components/Grid/GridContainer.jsx";
import Button from "components/CustomButtons/Button.jsx";
import Card from "components/Card/Card.jsx";
import CardHeader from "components/Card/CardHeader.jsx";
import CardBody from "components/Card/CardBody.jsx";
import CardFooter from "components/Card/CardFooter.jsx";

import { CircularProgress } from "@material-ui/core";

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
  }
};

class TrainJobs extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      "app":"fashion_mnist_app",
      "task": "IMAGE_CLASSIFICATION",
      "train_dataset_uri":"https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true",
      "test_dataset_uri":"https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true",
      "budget": { "MODEL_TRIAL_COUNT": 2 },
      "models": ["TfFeedForward"],
      "submit_status": "none" // "none", "submitting", "success", "failed"
    }
    this.handleClickButton = this.handleClickButton.bind(this)
  }

  handleInputChange = (event) => {
    const target = event.target;
    const value = target.type === 'checkbox' ? target.checked : target.value;
    const name = target.name;
    if (name === "modelName") { // need to make it an array
        this.setState({"models": target.value.split(",") })
    } else {
      this.setState({
        [name]: value
      });
    }
  }

  async handleClickButton(event) {
    event.preventDefault();
    debugger;
    const { appUtils: { rafikiClient, showError } } = this.props;

    try {
      console.log("button clicked")
      this.setState({ "submit_status":"submitting" })
      const json_params = { 
        "app": this.state.app,
        "task": this.state.task, 
        "train_dataset_uri": this.state.train_dataset_uri,
        "test_dataset_uri": this.state.test_dataset_uri,
        "budget": this.state.budget,
        "models": this.state.models
      }
      await rafikiClient.createTrainJob(json_params)
      this.setState({ "submit_status":"succeed" })
    } catch (error) {
      this.setState({ "submit_status": "failed" })
      showError(error, 'Failed createJobs');
    }
  }

  render() {
    /* const models = ["SkDt_RY0DI2PV26UMPJ2L", "TfFeedForward_GOZ9JANFQVVQT6LX",
    "SkDt_GOZ9JANFQVVQT6LX"
    ] */
    
    let SubmitButton = null
    
    switch (this.state.submit_status) {
      case "submitting":
        SubmitButton = <CircularProgress />
        break;
      case "succeed":
        SubmitButton = (<Link to="/admin/jobs">
          <Button color="primary">Back</Button>
        </Link>)
        break;
      default:
        SubmitButton = (
          <Button onClick={this.handleClickButton} color="primary">Create Jobs</Button>
        )
      break;
    }

    const { classes } = this.props;

    return (
    <div>
      <GridContainer justify="center" alignContent="center">
        <GridItem xs={12} sm={12} md={12} lg={12} xl={12}>
          <Card>
            <CardHeader color="primary">
              <h4 className={classes.cardTitleWhite}><Icon>add_photo_alternate</Icon>New Train Jobs</h4>
              <p className={classes.cardCategoryWhite}>Image Claasification</p>
            </CardHeader>
            <CardBody>
              <GridContainer justify="center" alignContent="center">
                <GridItem xs={12} sm={12} md={12} lg={8} xl={6}>
                    <TextField
                      name="app"
                      id="standard-full-width"
                      label="App"
                      onChange={this.handleInputChange}
                      placeholder="fashion_mnist_app"
                      helperText="Application Name"
                      fullWidth
                      margin="normal"
                    />
                    <TextField
                      name="task"
                      onChange={this.handleInputChange}
                      id="standard-full-width"
                      label="Task"
                      defaultValue="IMAGE_CLASSIFICATION"
                      placeholder="IMAGE_CLASSIFICATION"
                      fullWidth
                      disabled
                      margin="normal"
                    />
                    <TextField
                      name="train_dataset_uri"
                      onChange={this.handleInputChange}
                      id="standard-full-width"
                      label="Train Dataset URI"
                      defaultValue="https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true"
                      placeholder="https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true"
                      fullWidth
                      margin="normal"
                    />
                    <TextField
                      name="test_dataset_uri"
                      onChange={this.handleInputChange}
                      id="test_dataset_uri"
                      label="Test Dataset URI"
                      defaultValue="https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true"
                      placeholder="https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true"
                      fullWidth
                      margin="normal"
                    />
                     <TextField
                      name="modelName"
                      onChange={this.handleInputChange}
                      id="models"
                      label="Models"
                      helperText="Separate with ',' for mutiple models"
                      defaultValue="TfFeedForward"
                      placeholder="TfFeedForward"
                      fullWidth
                      margin="normal"
                    />
                </GridItem>
              </GridContainer>
              <GridContainer justify="center" alignContent="center">
                <GridItem xs={12} sm={12} md={12} lg={8} xl={6}>
               
                </GridItem>
              </GridContainer>
            </CardBody>
            <CardFooter>
              {SubmitButton}
            </CardFooter>
          </Card>
        </GridItem>
      </GridContainer>
    </div>
  );
           }
}

export default withStyles(styles)(TrainJobs);
