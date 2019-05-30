import React from "react";
import { Link } from "react-router-dom";

// @material-ui/core components
import withStyles from "@material-ui/core/styles/withStyles";
import Icon from "@material-ui/core/Icon";
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

function NewDatasets(props) {
  const { classes } = props;
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
              <GridContainer>
                <GridItem xs={12} sm={12} md={12} lg={8} xl={6}>
                  <CustomInput
                    labelText="Datasets Name"
                    id="company-disabled"
                    formControlProps={{
                      fullWidth: true
                    }}
                    inputProps={{
                      disabled: false
                    }}
                  />
                  <FormHelperText id="component-helper-text">*The datasets file must be in <b>.zip</b> archive format. inside which each sub-folder corresponds to a category. The train & test dataset's images should have the <b>same dimensions W x H</b>, accepted formats are  <b>JPEG or PNG</b></FormHelperText>
                </GridItem>
              </GridContainer>
              <GridContainer>
                <GridItem xs={12} sm={12} md={12}>
                  <p className={classes.cardCategory}>Upload from your computer</p>
                  <label htmlFor="outlined-button-file">
                    <Button variant="outlined" component="span" className={classes.button}>
                      Import Data
                    </Button>
                  </label>
                </GridItem>
              </GridContainer>
              <GridContainer>
                <GridItem xs={12} sm={12} md={12}>
                <h6 className={classes.cardCategory}>Data Distributuion</h6>
                <FormHelperText id="component-helper-text">*On Traning, Rafiki system will set aside 5% for validation</FormHelperText>
                </GridItem>
              </GridContainer>
            </CardBody>
            <CardFooter>
              <Link to="/admin/datasets">
              <Button color="info">Create New Datasets</Button>
              </Link>
            </CardFooter>
          </Card>
        </GridItem>
      </GridContainer>
    </div>
  );
}

export default withStyles(styles)(NewDatasets);
